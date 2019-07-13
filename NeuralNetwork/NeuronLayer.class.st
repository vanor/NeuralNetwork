"
This class represent a fully connected layer of neurons.

This class take some inputs and compute outputs to pass to  the next layer, and backpropagate the error to previous layer.

In regard of Authors of the book agileArtificialIntelligence. 

Public API and Key Messages

- message one   
- message two 
- (for bonus points) how to create instances.

   One simple example is simply gorgeous.
 
Internal Representation and Key Implementation Points.

    Instance Variables
	neurons:		<Object>
	nextLayer:		<Object>
	previousLayer:		<Object>


    Implementation Points
"
Class {
	#name : #NeuronLayer,
	#superclass : #Object,
	#instVars : [
		'previousLayer',
		'nextLayer',
		'neurons'
	],
	#category : #NeuralNetwork
}

{ #category : #initialization }
NeuronLayer >> backwardPropagateError [
	"This allow to calculate deltas for hidden layers. This is a recursive method"
	"We are in an hidden layer."
	
	neurons doWithIndex: [ :neuron :j |
		| theError |
		theError := 0.0.
		self nextLayer neurons do: [ : nextNeuron |
			theError := theError + ((nextNeuron weights at: j) * nextNeuron delta) ].
		neuron adjustDeltaWith: theError
	].

	self previousLayer notNil ifTrue: [ self previousLayer backwardPropagateError ].
]

{ #category : #initialization }
NeuronLayer >> backwardPropagateError: expected [
	"This allow to calculate deltas for the output layer since we need the expected output values."
	"We are in the output layer."
	
	neurons with: expected do: [ :neuron :exp |
		| theError |
		theError := exp - neuron output.
		neuron adjustDeltaWith: theError ].
	
	"We iterate on previous layers"
	self previousLayer notNil ifTrue: [ self previousLayer backwardPropagateError ].
]

{ #category : #'as yet unclassified' }
NeuronLayer >> deltas [
	"To get deltas."

	| depth sumDelta |
	depth := neurons anyOne weights size.
	^ (1 to: depth) collect: [ :j |
		sumDelta := 0.0.
		neurons do: [ :neuron |
			sumDelta := (neuron weights at: j) * (neuron delta) + sumDelta ].
		sumDelta ]
]

{ #category : #initialization }
NeuronLayer >> feed: someInputValues [
	"To feed the neuron layer with some inputs."
	
	| someOutputs |
	someOutputs := neurons collect: [ :n | n feed: someInputValues ].
	^ self isOutputLayer ifTrue: [ someOutputs ] ifFalse: [ nextLayer feed: someOutputs ]
]

{ #category : #initialization }
NeuronLayer >> initializeNbOfNeurons: nbOfNeurons nbOfWeights: nbOfWeights using: random [
	"Main method to initialize a neuron layer.
	nbOfNeurons: number of neurons the layer should be made of.
	nbOfWeights: number of weights each neuron should have.
	random: a random number generator."

	| weights |
	neurons := (1 to: nbOfNeurons) collect: [ :i |
		weights := (1 to: nbOfWeights) collect: [ :ii | random next * 4 - 2 ].
		Neuron new sigmoid; weights: weights; bias: (random next * 4 - 2) ].
	self learningRate: 0.1
]

{ #category : #testing }
NeuronLayer >> isFullyConnected [
	"To know if the layer is a fully connected layer."

	^ true
]

{ #category : #initialization }
NeuronLayer >> isOutputLayer [
	"Return true if the layer is the output layer, i.e. the last layer."
	
	^ self nextLayer isNil
]

{ #category : #initialization }
NeuronLayer >> learningRate: aLearningRate [
	"Set the learning rate for all the neurons.
	Note that this method should be called after configuring the network, and _not_ before."

	self assert: [ neurons notEmpty ] description: 'learningRate: should be invoked after configuring the layer'.
	neurons do: [ :n | n learningRate: aLearningRate ]
]

{ #category : #initialization }
NeuronLayer >> neurons [
	"Get the neurons i am composed of."
	
	^ neurons
]

{ #category : #initialization }
NeuronLayer >> nextLayer [
	"Get the next layer connected to me."
	
	^ nextLayer
]

{ #category : #initialization }
NeuronLayer >> nextLayer: aLayer [
	"Set the next layer."
	
	nextLayer := aLayer
]

{ #category : #initialization }
NeuronLayer >> numberOfNeurons [
	"Get the number of neurons i am composed of."
	
	^ neurons size
]

{ #category : #initialization }
NeuronLayer >> previousLayer [
	"Get the previous layer connected to me."
	
	^ previousLayer
]

{ #category : #initialization }
NeuronLayer >> previousLayer: aLayer [
	"Set the previous layer."
	
	previousLayer := aLayer
]

{ #category : #initialization }
NeuronLayer >> updateWeights [
	"To update weights of a neuron based the set of previous layer outputs as input. Then, we are in the second hidden layer or in the output layer."
	
	| inputs |
	self previousLayer isFullyConnected ifTrue: [ inputs := self previousLayer neurons collect: #output ]
								 ifFalse: [ inputs := self previousLayer outputs flattened ].
	
	self updateWeights: inputs
]

{ #category : #initialization }
NeuronLayer >> updateWeights: initialInputs [
	"To update weights of a neuron based the set of initial input. Then, we are in the first hidden layer."
	
	| inputs |
	inputs := initialInputs.
	
	neurons do: [ :n |
		n adjustWeightsWithInput: inputs.
		n adjustBias ].
	
	self nextLayer ifNotNil: [ self nextLayer updateWeights ]
]
