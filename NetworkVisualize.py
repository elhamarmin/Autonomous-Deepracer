import curses
import os
import subprocess
import time
import torch
from watchdog.events import FileSystemEventHandler
import matplotlib.pyplot as plt

from Policy_Network import Policy_Network

class FileHandler(FileSystemEventHandler):
    def __init__(self, plot_function):
        self.plot_function = plot_function

    def on_created(self, event):
        if event.is_directory:
            return
        self.plot_function()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

last_file = ""
index_file = -1

def read_and_plot(directory, command):
    global last_file,index_file
    
    files = sorted([f for f in os.listdir(directory) if f.startswith('ep_vis_')], key=lambda x: int(x[7:]))

    if len(files) == 0:
        last_file = ''
        return
    
    if command == 0:
        index_file = len(files) - 1
    elif command == 1:
        if index_file + 1 <= len(files) - 1:
            index_file = index_file + 1
    elif command == -1:
        if index_file - 1 >= 0:
            index_file = index_file - 1
    
    newest_file = files[index_file]
    
    if not last_file == newest_file:
        last_file = newest_file
        print(last_file)
        network = Policy_Network(input_size=3, hidden_size=6, output_size=3).to(device)
        network.load_state_dict(torch.load(f'{directory}/{last_file}'))
        plt.clf()
        network.draw_graph()
        
def main(stdscr):
    # Set up the screen
    curses.curs_set(0)  # Hide cursor
    stdscr.nodelay(True)  # Make getch non-blocking
    stdscr.clear()
    stdscr.refresh()

    directory_to_watch = "./results"

    figure, ax = plt.subplots(figsize=(10, 8))
    
    # Initial plot
    plt.ion()  # Turn on interactive mode
    plt.figure()
    
    command = 0
    
    # Main loop
    while True:
        stdscr.refresh()
        stdscr.nodelay(False)  # Make getch blocking
        stdscr.clear()
        
        stdscr.addstr(0, 0, "Enter Command (Left Arrow , Right Arrow)")
        stdscr.refresh()
        
        print(last_file)

        stdscr.timeout(5000)  # Set timeout to 5 seconds (in milliseconds)

        try:
            key = stdscr.getch()
        except curses.error:  # If timeout occurs
            command = 0

        # Log left arrow key pressed
        if key == curses.KEY_LEFT:
            command = -1
        if key == curses.KEY_RIGHT:
            command = 1
            
        print(command)

        read_and_plot(directory_to_watch , command)
        
        plt.title("Neural Network Visualization")
        figure.canvas.draw()
        figure.canvas.flush_events()
        
        # Move based on user input
        if key == ord('q'):
            break  # Quit if 'q' is pressed

if __name__ == "__main__":

    import webbrowser
    subprocess.Popen(["tensorboard", "--logdir", "logs/train"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    webbrowser.open('http://localhost:6006')  # Go to example.com

    curses.wrapper(main)
  