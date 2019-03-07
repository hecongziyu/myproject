一、base operator
1、分类，分别按price、volume等进行分类
2、


二、machine learning
1、SeqFinace



三、FinaceGan
1、target lstm model -->  gen real data, need pretrained by real data. out put: action(3) probs
   
   ??? ?????   target lstm cant generator real data, check seqgan lstm how to generator real data

2、generator model --> input: trans history . out put : action (3) probs
3、d model :
        input -- > target lstm  --> seq trans signal: true
        input -->  generator m  --> seq trans signal: false

4、rl model:
        input --> g model , target, d model
              --> gmodel ouput - d model --> reward
              --> loss = -sum((gmodel mask target) * reward)   (note: target value is 0 or 1)
              --> loss backward()
              --> update g mode parameters

ref :
SegGan
1、target lstm model --> input: [seq][edb]  out put : seq prob   根据真实数据预训练出的模型，可以生成真实数据
2、generator model --> input: seq sentents or zero. --> log softmax -->   seq
3、rl model:
        A. params: 1)  ori model(gen model) 2) own model (copy ori model)
        for(l : range(1:seq len)){
            sample = own model gen seq ( lenth = l)
            pred = d model return
            reward[l - 1] += pred
        }

        return reward/num





3、d model --> input seq is true or false
loss: 
1) rl model loss:
	prob action predict - target action predict
	value predict - target value
	
								mtcs 1 		reward(d model)
	target action predict ---> 	mtcs 2		reward(d model)		--->	
								mtcs ... 	reward(d model)

	