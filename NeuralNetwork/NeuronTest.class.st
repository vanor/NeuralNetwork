"
This class is for unit test the Neuron class as a sigmoid neuron.

For add methods for test use cases of Neuron class.

In regard of authors of the book.  

Public API and Key Messages

- message one   
- message two 
- (for bonus points) how to create instances.

   One simple example is simply gorgeous.
 
Internal Representation and Key Implementation Points.


    Implementation Points
"
Class {
	#name : #NeuronTest,
	#superclass : #TestCase,
	#category : #NeuralNetwork
}

{ #category : #tests }
NeuronTest >> testTrainingAND [
	"To test if a sigmoid neuron learn to resolve AND problem."

	| p |
	p := Neuron new.
	p weights: #(-1 -1).
	p bias: 2.
	
	5000 timesRepeat: [ 
		p train: #(0 0) desiredOutput: 0.
		p train: #(0 1) desiredOutput: 0.
		p train: #(1 0) desiredOutput: 0.
		p train: #(1 1) desiredOutput: 1 ].
	
	self assert: ((p feed: #(0 0)) closeTo: 0 precision: 0.1).
	self assert: ((p feed: #(0 1)) closeTo: 0 precision: 0.1).
	self assert: ((p feed: #(1 0)) closeTo: 0 precision: 0.1).
	self assert: ((p feed: #(1 1)) closeTo: 1 precision: 0.1).
]

{ #category : #tests }
NeuronTest >> testTrainingNAND [
	"To test if a sigmoid neuron learn to resolve NAND problem."

	| p |
	p := Neuron new.
	p weights: #(-1 -1).
	p bias: 2.
	
	5000 timesRepeat: [ 
		p train: #(0 0) desiredOutput: 1.
		p train: #(0 1) desiredOutput: 1.
		p train: #(1 0) desiredOutput: 1.
		p train: #(1 1) desiredOutput: 0 ].
	
	self assert: ((p feed: #(0 0)) closeTo: 1 precision: 0.1).
	self assert: ((p feed: #(0 1)) closeTo: 1 precision: 0.1).
	self assert: ((p feed: #(1 0)) closeTo: 1 precision: 0.1).
	self assert: ((p feed: #(1 1)) closeTo: 0 precision: 0.1).
]

{ #category : #tests }
NeuronTest >> testTrainingNOR [
	"To test if a sigmoid neuron learn to resolve NOR problem."

	| p |
	p := Neuron new.
	p weights: #(-1 -1).
	p bias: 2.
	
	5000 timesRepeat: [ 
		p train: #(0 0) desiredOutput: 1.
		p train: #(0 1) desiredOutput: 0.
		p train: #(1 0) desiredOutput: 0.
		p train: #(1 1) desiredOutput: 0 ].
	
	self assert: ((p feed: #(0 0)) closeTo: 1 precision: 0.1).
	self assert: ((p feed: #(0 1)) closeTo: 0 precision: 0.1).
	self assert: ((p feed: #(1 0)) closeTo: 0 precision: 0.1).
	self assert: ((p feed: #(1 1)) closeTo: 0 precision: 0.1).
]

{ #category : #tests }
NeuronTest >> testTrainingNOT [
	"To test if a sigmoid neuron learn to resolve NOT problem."

	| p |
	p := Neuron new.
	p weights: #(-1).
	p bias: 2.
	
	5000 timesRepeat: [ 
		p train: #(0) desiredOutput: 1.
		p train: #(1) desiredOutput: 0 ].
	
	self assert: ((p feed: #(0)) closeTo: 1 precision: 0.1).
	self assert: ((p feed: #(1)) closeTo: 0 precision: 0.1).
]

{ #category : #tests }
NeuronTest >> testTrainingOR [
	"To test if a sigmoid neuron learn to resolve OR problem."

	| p |
	p := Neuron new.
	p weights: #(-1 -1).
	p bias: 2.
	
	5000 timesRepeat: [ 
		p train: #(0 0) desiredOutput: 0.
		p train: #(0 1) desiredOutput: 1.
		p train: #(1 0) desiredOutput: 1.
		p train: #(1 1) desiredOutput: 1 ].
	
	self assert: ((p feed: #(0 0)) closeTo: 0 precision: 0.1).
	self assert: ((p feed: #(0 1)) closeTo: 1 precision: 0.1).
	self assert: ((p feed: #(1 0)) closeTo: 1 precision: 0.1).
	self assert: ((p feed: #(1 1)) closeTo: 1 precision: 0.1).
]
