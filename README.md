-------------------------------------------------------

**Code repository for ERA V1 Assignment 6: Backpropagation & Architectural Basics**

-------------------------------------------------------

**PART 1**

For the first part of the assignment, the task is to determine the total loss using the backpropagation method. The neural network employed is a simple one, consisting of 2 inputs, 1 hidden layer, and 1 output layer. The output layer comprises 2 categories.

Within this architecture, there are a total of 8 weights that need to be calculated to minimize the loss. Each node incorporates an activation function, which generates the output based on the given target. In this particular case, the sigmoid activation function (1 / (1 + e^-x)) has been utilized.

The backpropagation method requires not only but also the gradients of total error with the weights. This is calulated in order to update the weights in the next step.  

The gradients are calculated by converting them to a algebraic equation.

**First Set of Formula** <br />
h1 = w1*i1 + w2*i2  <br />                
h2 = w3*i1 + w4*i2  <br />
a_h1 = σ(h1) = 1/(1 + exp(-h1))  <br />
a_h2 = σ(h2)           <br />
o1 = w5*a_h1 + w6*a_h2  <br />
o2 = w7*a_h1 + w8*a_h2  <br />
a_o1 = σ(o1)          <br />
a_o2 = σ(o2)          <br />
E_total = E1 + E2     <br />
E1 = ½ * (t1 - a_o1)²  <br />
E2 = ½ * (t2 - a_o2)²	<br />	

**Second Set of Formula**<br />
∂E_total/∂w5 = ∂(E1 + E2)/∂w5<br />					
∂E_total/∂w5 = ∂E1/∂w5		<br />			
∂E_total/∂w5 = ∂E1/∂w5 = ∂E1/∂a_o1*∂a_o1/∂o1*∂o1/∂w5<br />					
∂E1/∂a_o1 =  ∂(½ * (t1 - a_o1)²)/∂a_o1 = (a_01 - t1)	<br />				
∂a_o1/∂o1 =  ∂(σ(o1))/∂o1 = a_o1 * (1 - a_o1)		<br />			
∂o1/∂w5 = a_h1		<br />

**Third Set of Formula**   <br />
**∂E_total/∂w5 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h1	**	<br />			
**∂E_total/∂w6 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h2	**	<br />			
**∂E_total/∂w7 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h1	**	<br />			
**∂E_total/∂w8 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h2	**      <br />

**Fourth Set of Formula**				<br />
∂E1/∂a_h1 = (a_01 - t1) * a_o1 * (1 - a_o1) * w5	<br />							
∂E2/∂a_h1 = (a_02 - t2) * a_o2 * (1 - a_o2) * w7	<br />							
∂E_total/∂a_h1 = (a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7   <br />								
∂E_total/∂a_h2 = (a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8	<br />	


**Fifth Set of Formula**	<br />
∂E_total/∂w1 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1    <br />					
∂E_total/∂w2 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w2	<br />				
∂E_total/∂w3 = ∂E_total/∂a_h2 * ∂a_h2/∂h2 * ∂h2/∂w3	<br />				
						

**Sixth Set of Formula**   <br />
**∂E_total/∂w1 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1    **	<br />											
**∂E_total/∂w2 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2    **	<br />											
**∂E_total/∂w3 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i1    **	<br />											
**∂E_total/∂w4 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i2    **<br />


** The Third and Sixth set of Formula are that are used for backward propagation**												

			
** Summarazing it also as chatgpt says:**

The backpropagation algorithm is used to train neural networks by adjusting the weights and biases based on the calculated gradients of the loss function. Here are the general steps involved in the backpropagation algorithm:

1. Initialize Weights and Biases: Start by assigning random values to the weights and biases in the neural network.

2. Forward Propagation: Perform a forward pass through the neural network to calculate the predicted outputs. This involves multiplying the inputs by the weights, applying the activation function, and passing the results to the next layer until the final output is obtained.

3. Calculate Loss: Compare the predicted outputs with the actual targets and calculate the loss using a suitable loss function (e.g., mean squared error or cross-entropy).

4. Backward Propagation: Compute the gradients of the loss function with respect to the weights and biases by applying the chain rule. Start from the output layer and move backward, calculating the gradients at each layer.

5. Update Weights and Biases: Adjust the weights and biases using an optimization algorithm (e.g., gradient descent) based on the calculated gradients. This step aims to minimize the loss by iteratively updating the weights and biases.

6. Repeat: Repeat steps 2 to 5 for a specified number of epochs or until the desired level of accuracy or convergence is achieved.

7. Evaluate the Trained Model: Once the training process is complete, evaluate the performance of the trained model on a separate test set to assess its generalization ability.


The below image is screenshot for the excel practice done. I have used the same excel sheet and written the formula again. Current loss is shown for learning rate of 2.0. I have extended the table to 168 steps to understand the nature of curve in more detail. 
![alt text](https://github.com/saurabhmangal/era1_s6/blob/master/s6_excel_ss_backpropogation.JPG)


The below image is pattern of the total loss with respect to each step for different values of learning rate. The range used is 0.1 to  2.0. It is observed that higher learning rate for this range converges faster.

<img src="https://github.com/saurabhmangal/era1_s6/blob/master/E_total_vs_Learning_rate.png" alt="alt text" width="600px">

-------------------------------------------------------
-------------------------------------------------------

**PART 2**




