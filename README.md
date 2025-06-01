both single apple spawn and multi apple spawn models trained on following specs:
- 400x400 grid
- single reward or multi reward apple spawn at a time 
- trained on NVIDIA RTX 6000 Ada Generation Graphics Card 

*note: model was tested on a 640x640 grid and generalized well. larger grid was used for better demonstration.

model training visualizer (example from multi apple spawn training):



https://github.com/user-attachments/assets/ebf02580-a875-471e-ac04-c11999cd2547




following video shows a demo of the snake_dqn.pth model (single apple spawn) in action:

https://github.com/user-attachments/assets/af64c5b4-5b18-428a-9671-78d7f2de9134

following video shows a demo of the snake_dqn_multi_path_reward.pth (multi apple spawn) in action:

https://github.com/user-attachments/assets/3fad507e-444d-4c17-bf7c-2462fa626721

this struggles big time with accurate pathfinding (need to implement a better pathfinding algorithm). maybe more episodes of training might help? 




(made as a part of internal internship presentation. thank you to all the advisers. ofc thank you perplexity pro + github copilot for debugging help besties.)
