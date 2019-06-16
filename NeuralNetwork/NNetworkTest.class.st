"
This class is for unit test the class NNetwork.

In this class, we have written the tests of some common use cases of Neural Networks.

In regard of Authors of the book. 

Public API and Key Messages

- message one   
- message two 
- (for bonus points) how to create instances.

   One simple example is simply gorgeous.
 
Internal Representation and Key Implementation Points.


    Implementation Points
"
Class {
	#name : #NNetworkTest,
	#superclass : #TestCase,
	#category : #NeuralNetwork
}

{ #category : #tests }
NNetworkTest >> testBasic [
	"To test the basic behavior of a neural network."

	| n |
	n := NNetwork new.
	n configure: 2 hidden: 2 nbOfOutputs: 1.
	self assert: ((n feed: #(1 2)) anyOne closeTo: 0.6658967122703221 precision: 0.0000000001).
	self assert: n numberOfOutputs equals: 1.
]

{ #category : #tests }
NNetworkTest >> testXOR [
	"To test if a neural network lear to resolve XOR problem."

	| n |
	n := NNetwork new.
	n configure: 2 hidden: 3 nbOfOutputs: 1.
	
	20000 timesRepeat: [ 
		n train: #(0 0) desiredOutputs: #(0).
		n train: #(0 1) desiredOutputs: #(1).
		n train: #(1 0) desiredOutputs: #(1).
		n train: #(1 1) desiredOutputs: #(0) ].
	
	self assert: (n feed: #(0 0)) first < 0.1.
	self assert: (n feed: #(0 1)) first > 0.9.
	self assert: (n feed: #(1 0)) first > 0.9.
	self assert: (n feed: #(1 1)) first < 0.1.
]
