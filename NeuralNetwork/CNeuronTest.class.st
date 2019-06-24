"
This class allow to test the behavior of a neuron in a convolutional neural network.

There, we just implement few basic test with features and bias.

No collaborators for now. 

Public API and Key Messages

- message one   
- message two 
- (for bonus points) how to create instances.

   One simple example is simply gorgeous.
 
Internal Representation and Key Implementation Points.


    Implementation Points
"
Class {
	#name : #CNeuronTest,
	#superclass : #TestCase,
	#category : #NeuralNetwork
}

{ #category : #tests }
CNeuronTest >> testBasic [
	"To test if a Neuron can operate on volumetric inputs and weights, like a convolutional neuron do."

	| cuboid1 cuboid2 arr n |
	cuboid1 := #(
					#(#(0 0 0) #(0 0 0) #(0 0 0))
					#(#(0 0 0) #(0 1 2) #(2 0 1))
					#(#(0 0 0) #(0 2 2) #(0 1 1))).
	cuboid2 := #(
					#(#(1 0 -1) #(0 1 1) #(-1 0 1))
					#(#(1 -1 1) #(1 1 0) #(0 -1 1))
					#(#(-1 1 0) #(0 -1 1) #(-1 -1 -1))).
	n := Neuron new.
	
]
