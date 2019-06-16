"
This class is for unit test the Neuron class as a Perceptron.

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
	#name : #PerceptronTest,
	#superclass : #TestCase,
	#category : #NeuralNetwork
}

{ #category : #'as yet unclassified' }
PerceptronTest >> digitalComparator: inputs [
	"Return an array of three elements"

	| not and nor a b aGb aEb aLb notA notB |
	a := inputs first.
	b := inputs second.
	
	and := Neuron new weights: #(1 1); bias: -1.5; step.
	not := Neuron new weights: #(-1); bias: 0.5; step.
	nor := Neuron new weights: #(-1 -1); bias: 0.5; step.
	
	notA := not feed: { a }.
	notB := not feed: { b }.
	
	aLb := and feed: { notA . b }.
	aGb := and feed: { a . notB }.
	aEb := nor feed: { aGb . aLb }.
	^ { aGb . aEb . aLb }
]

{ #category : #tests }
PerceptronTest >> testAND [
	"To test AND logical gate for an implementation with a perceptron"

	| p |
	p := Neuron new.
	p step.
	p weights: #(1 1).
	p bias: -1.5.
	
	self assert: (p feed: #(0 0)) equals: 0.
	self assert: (p feed: #(0 1)) equals: 0.
	self assert: (p feed: #(1 0)) equals: 0.
	self assert: (p feed: #(1 1)) equals: 1.
]

{ #category : #tests }
PerceptronTest >> testDigitalComparator [
	"For test the method PerceptronTest>>digitalComparator"

	self assert: (self digitalComparator: #(0 0)) equals: #(0 1 0).
	self assert: (self digitalComparator: #(0 1)) equals: #(0 0 1).
	self assert: (self digitalComparator: #(1 0)) equals: #(1 0 0).
	self assert: (self digitalComparator: #(1 1)) equals: #(0 1 0).
]

{ #category : #tests }
PerceptronTest >> testNAND [
	"To test NAND logical gate for an implementation with a perceptron"

	| p |
	p := Neuron new.
	p step.
	p weights: #(-1 -1).
	p bias: 1.5.
	
	self assert: (p feed: #(0 0)) equals: 1.
	self assert: (p feed: #(0 1)) equals: 1.
	self assert: (p feed: #(1 0)) equals: 1.
	self assert: (p feed: #(1 1)) equals: 0.
]

{ #category : #tests }
PerceptronTest >> testNOR [
	"To test NOR logical gate for an implementation with a perceptron"

	| p |
	p := Neuron new.
	p step.
	p weights: #(-1 -1).
	p bias: 0.5.
	
	self assert: (p feed: #(0 0)) equals: 1.
	self assert: (p feed: #(0 1)) equals: 0.
	self assert: (p feed: #(1 0)) equals: 0.
	self assert: (p feed: #(1 1)) equals: 0.
]

{ #category : #tests }
PerceptronTest >> testNOT [
	"To test NOT logical gate for an implementation with a perceptron"

	| p |
	p := Neuron new.
	p step.
	p weights: #(-1).
	p bias: 0.5.
	
	self assert: (p feed: #(0)) equals: 1.
	self assert: (p feed: #(1)) equals: 0.
]

{ #category : #tests }
PerceptronTest >> testOR [
	"To test OR logical gate for an implementation with a perceptron"

	| p |
	p := Neuron new.
	p step.
	p weights: #(1 1).
	p bias: -0.5.
	
	self assert: (p feed: #(0 0)) equals: 0.
	self assert: (p feed: #(0 1)) equals: 1.
	self assert: (p feed: #(1 0)) equals: 1.
	self assert: (p feed: #(1 1)) equals: 1.
]

{ #category : #tests }
PerceptronTest >> testSmallExample [
	"To test if the Neuron produce the desired output for a specific input"

	| p result |
	p := Neuron new.
	p step.
	p weights: #(1 2).
	p bias: -2.
	result := p feed: #(5 2).
	self assert: result equals: 1.
	
	result := p feed: #(-2 2).
	self assert: result equals: 0.
]

{ #category : #'as yet unclassified' }
PerceptronTest >> testTrainingAND [
	"To test if a perceptron with arbitrary weights learn to resolve AND problem"

	| p |
	p := Neuron new.
	p step.
	p weights: #(-1 -1).
	p bias: 2.
	
	40 timesRepeat: [ 
		p train: #(0 0) desiredOutput: 0.
		p train: #(0 1) desiredOutput: 0.
		p train: #(1 0) desiredOutput: 0.
		p train: #(1 1) desiredOutput: 1. ].
	
	self assert: (p feed: #(0 0)) equals: 0.
	self assert: (p feed: #(0 1)) equals: 0.
	self assert: (p feed: #(1 0)) equals: 0.
	self assert: (p feed: #(1 1)) equals: 1.
]

{ #category : #'as yet unclassified' }
PerceptronTest >> testTrainingNAND [
	"To test if a perceptron with arbitrary weights learn to resolve NAND problem"

	| p |
	p := Neuron new.
	p step.
	p weights: #(-1 -1).
	p bias: 2.
	
	40 timesRepeat: [ 
		p train: #(0 0) desiredOutput: 1.
		p train: #(0 1) desiredOutput: 1.
		p train: #(1 0) desiredOutput: 1.
		p train: #(1 1) desiredOutput: 0. ].
	
	self assert: (p feed: #(0 0)) equals: 1.
	self assert: (p feed: #(0 1)) equals: 1.
	self assert: (p feed: #(1 0)) equals: 1.
	self assert: (p feed: #(1 1)) equals: 0.
]

{ #category : #'as yet unclassified' }
PerceptronTest >> testTrainingNOR [
	"To test if a perceptron with arbitrary weights learn to resolve NOR problem"

	| p |
	p := Neuron new.
	p step.
	p weights: #(-1 -1).
	p bias: 2.
	
	40 timesRepeat: [ 
		p train: #(0 0) desiredOutput: 1.
		p train: #(0 1) desiredOutput: 0.
		p train: #(1 0) desiredOutput: 0.
		p train: #(1 1) desiredOutput: 0. ].
	
	self assert: (p feed: #(0 0)) equals: 1.
	self assert: (p feed: #(0 1)) equals: 0.
	self assert: (p feed: #(1 0)) equals: 0.
	self assert: (p feed: #(1 1)) equals: 0.
]

{ #category : #'as yet unclassified' }
PerceptronTest >> testTrainingNOT [
	"To test if a perceptron with arbitrary weights learn to resolve NOT problem"

	| p |
	p := Neuron new.
	p step.
	p weights: #(-1).
	p bias: 2.
	
	40 timesRepeat: [ 
		p train: #(0) desiredOutput: 1.
		p train: #(1) desiredOutput: 0. ].
	
	self assert: (p feed: #(0)) equals: 1.
	self assert: (p feed: #(1)) equals: 0.
]

{ #category : #'as yet unclassified' }
PerceptronTest >> testTrainingOR [
	"To test if a perceptron with arbitrary weights learn to resolve OR problem"

	| p |
	p := Neuron new.
	p step.
	p weights: #(-1 -1).
	p bias: 2.
	
	40 timesRepeat: [ 
		p train: #(0 0) desiredOutput: 0.
		p train: #(0 1) desiredOutput: 1.
		p train: #(1 0) desiredOutput: 1.
		p train: #(1 1) desiredOutput: 1. ].
	
	self assert: (p feed: #(0 0)) equals: 0.
	self assert: (p feed: #(0 1)) equals: 1.
	self assert: (p feed: #(1 0)) equals: 1.
	self assert: (p feed: #(1 1)) equals: 1.
]

{ #category : #tests }
PerceptronTest >> testWrongFeeding [
	"To test if inputs have the same size than weights"

	| p |
	p := Neuron new.
	p weights: #(-1).
	p bias: 0.5.
	
	self should: [ p feed: #(1 1) ] raise: Error.
]
