import time

from MacRacerTraining import MacRacerTraining
from WebotsEnvironment import WebotsEnvironment

run_model = 0 # 0 = Trian , 1 = Test , 2 = Debug    

if run_model == 0:
    
    macRacerTr = MacRacerTraining(train=True)
    # macRacerTr.manual_train_model(learning_rate=0.0003,total_timesteps=1024,ent_coef=0.15,clip_range=0.15,gae_lambda=0.78,
    #                     n_epochs=64,n_steps=256,batch_size=64)
    
    # obs = macRacerTr.env.reset()[0]
    
    macRacerTr.wandb_train()
    
elif run_model == 1:
    
    macRacerTr = MacRacerTraining(train=False)
    
    model_path = './trained_model/model_20240723-1425.zip'

    model = macRacerTr.load_model(model_path)
    
elif run_model == 2:
    env = WebotsEnvironment()
    
obs = [0,0] # [0] = Left Points and [1] = Right points Weight

while True:
    if run_model == 0:
        break
    
    if run_model == 1:
        val_obs = obs
        
        action, _states = model.predict(val_obs) # action[length = 2]: action[0] = forward/backward , action[1] = left/right
        
        print(f'Forward Value: {action[0]:.2f}  Steering Value (Rad): {action[1]:.2f}',end='\r')
        time.sleep(0.2)

    if run_model == 2:
        key = input("Enter Key:")
    
        if key == 'a':
            env.apply_action([0,-1])
        elif key == 'd':
            env.apply_action([0,1])
        elif key == 'w':
            env.apply_action([1,0])
        elif key == 's':
            env.apply_action([-1,0])
        elif key == 'i':
            env.robot.step(500)
            
        lidar_data = env.get_lidar_score()
        print(lidar_data)
        