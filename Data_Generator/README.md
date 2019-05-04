# The Environment of DRL
An implementation of the environment of Deep Reinforcement Learning in specific field.
<br/></br>
## The Trajectory Environment
A **trajectory environment** for **Mobile Crowdsensing Problem** solved by **DRL**. Including management of participants and tasks, feature extraction of trajectory factors, system state simulation and reward computing, etc.
<br/></br>
### 1. Mobile Crowdsensing Problem and DRL

With a massive deployment of mobile devices, crowdsourcing has become a new service paradigm in which a task requester could recruit a batch of participants with a mobile IoT device from our system for quick and accurate results. In a mobile industrial crowdsouring platform, a large amount of data is collected, extracted information and distributed to the requesters.(More information about mobile crowdsensing:https://en.wikipedia.org/wiki/Crowdsensing)

The participant selection problem in mobile Crowdsensing has been proved to be a NP-hard problem. We presents a multi-goal participant selection approach to iteratively update the participant selection policy via multi-task deep reinforcement learning.
<br/></br>
### 2. Trajectory Environment
We got two generators in the environment simulator combining real datasets to stimulate the participant selection process, with a Trajectory Manager, in charge of the management of participants and tasks. A policy algorithm gets all the features of participants and tasks from the States Simulator who synchronizes the system parameters and updates the whole state of the environment continuously and returns a list of appropriate actions which are run by the Action Processor. All the feedback rewards are evaluated by the Reward Computing module.

<div align=center><img src="https://github.com/emailhxn/Deep-Reinforcement-Learning/blob/master/Data_Generator/img/DRLArch03.png" width = "480" height = "320"/></div>

<!--
![](https://github.com/emailhxn/Deep-Reinforcement-Learning/blob/master/Data_Generator/img/DRLArch03.png)
-->

<br/></br>
### 3. Dataset
We got the raw dataset from this post: [Analyzing 1.1 Billion NYC Taxi and Uber Trips, with a Vengeance.][1]  

"An open-source exploration of the city's neighborhoods, nightlife, airport traffic, and more, through the lens of publicly available taxi and Uber data."

[GitHub link click here][2]

[1]: https://toddwschneider.com/posts/analyzing-1-1-billion-nyc-taxi-and-uber-trips-with-a-vengeance/  
[2]: https://github.com/toddwschneider/nyc-taxi-data

Only part of the raw data is used in our demo.
<div align=left><img src="https://github.com/emailhxn/Deep-Reinforcement-Learning/blob/master/Data_Generator/img/dataset-table.png" width = "280" height = "160"/></div>
