"
This class represent the network as a whole.

This class allow a user to create a neural network and feed it with some inputs, and backpropagate the error when training the network.

In regard of Authors of the book. 

Public API and Key Messages

- message one   
- message two 
- (for bonus points) how to create instances.

   One simple example is simply gorgeous.
 
Internal Representation and Key Implementation Points.

    Instance Variables
	errors:		<Object>
	layers:		<Object>
	precisions:		<Object>


    Implementation Points
"
Class {
	#name : #NNetwork,
	#superclass : #Object,
	#instVars : [
		'layers',
		'errors',
		'precisions'
	],
	#category : #NeuralNetwork
}

{ #category : #initialization }
NNetwork >> addLayer: aNeuronLayer [
	"To add a neural layer. The added layer is linked to the already added layers."

	layers ifNotEmpty: [ 
		aNeuronLayer previousLayer: layers last.
		layers last nextLayer: aNeuronLayer ].
	layers add: aNeuronLayer.
]

{ #category : #initialization }
NNetwork >> backwardPropagateError: expectedOutputs [
	"To propagate the error on expectedOutputs corresponding to the desired outputs on a trainig sample."

	self outputLayer backwardPropagateError: expectedOutputs
]

{ #category : #initialization }
NNetwork >> configure: nbOfInputs hidden: nbOfNeurons1 hidden: nbOfNeurons2 nbOfOutputs: nbOfOutputs [
	"To configure the network with two hidden and one output layer. The params specifies the size of layers."

	| random |
	random := Random seed: 42.
	self addLayer: (NeuronLayer new initializeNbOfNeurons: nbOfNeurons1 nbOfWeights: nbOfInputs using: random).
	self addLayer: (NeuronLayer new initializeNbOfNeurons: nbOfNeurons2 nbOfWeights: nbOfNeurons1 using: random).
	self addLayer: (NeuronLayer new initializeNbOfNeurons: nbOfOutputs nbOfWeights: nbOfNeurons2 using: random).
]

{ #category : #initialization }
NNetwork >> configure: nbOfInputs hidden: nbOfNeurons nbOfOutputs: nbOfOutputs [
	"To configure the network with one hidden and output layer. The params specifies the size of layers."

	| random |
	random := Random seed: 42.
	self addLayer: (NeuronLayer new initializeNbOfNeurons: nbOfNeurons nbOfWeights: nbOfInputs using: random).
	self addLayer: (NeuronLayer new initializeNbOfNeurons: nbOfOutputs nbOfWeights: nbOfNeurons using: random).
]

{ #category : #initialization }
NNetwork >> feed: someInputValues [
	"Feed the first layer with the provided inputs."

	^ layers first feed: someInputValues.
]

{ #category : #initialization }
NNetwork >> initialize [
	"To initialize the network with a configuration."

	super initialize.
	layers := OrderedCollection new.
	errors := OrderedCollection new.
	precisions := OrderedCollection new.
]

{ #category : #initialization }
NNetwork >> learningRate: aLearningRate [
	"To set the learning rate for all the layers."

	layers do: [ :layer | layer learningRate: aLearningRate ]
]

{ #category : #initialization }
NNetwork >> numberOfOutputs [
	"To get the number of output of the network."

	^ layers last numberOfNeurons
]

{ #category : #initialization }
NNetwork >> outputLayer [
	"To return the output layer, which is also the last layer."

	^ layers last
]

{ #category : #initialization }
NNetwork >> predict: inputs [
	"To make a prediction. This method assumes that the number of outputs is the same than the number of different values the network can output."

	| outputs |
	outputs := self feed: inputs.
	^ (outputs indexOf: (outputs max)) - 1
]

{ #category : #initialization }
NNetwork >> train: someInputs desiredOutputs: desiredOutputs [
	"To train the whole neural network with a set of inputs and some expected output."

	| realOutputs t |
	realOutputs := self feed: someInputs.
	t := (1 to: desiredOutputs size) collect: [ :i |
		((desiredOutputs at: i) - (realOutputs at: i )) raisedTo: 2 ].
	
	self backwardPropagateError: desiredOutputs.
	self updateWeights: someInputs.
]

{ #category : #initialization }
NNetwork >> train: trainingSet nbEpoch: nbOfEpoch [
	"To train the network using the training data set."

	| sumError outputs expectedOutputs epochPrecision t |
	1 to: nbOfEpoch do: [ :epoch |
		sumError := 0.
		epochPrecision := 0.
		trainingSet do: [ :row |
			outputs := self feed: row allButLast.
			expectedOutputs := (1 to: self numberOfOutputs) collect: [ 0 ].
			expectedOutputs at: (row last) +1 put: 1.
			(row last = (self predict: row allButLast)) ifTrue: [ epochPrecision := epochPrecision + 1 ].
			t := (1 to: expectedOutputs size)
						collect: [ :i | ((expectedOutputs at: i) - (outputs at: i)) raisedTo: 2 ].
			sumError := sumError + t sum.
			
			self backwardPropagateError: expectedOutputs.
			self updateWeights: row allButLast.
		].
		errors add: sumError.
		precisions add: (epochPrecision / trainingSet size) asFloat.
	]
]

{ #category : #initialization }
NNetwork >> updateWeights: initialInputs [
	"To update the weights of the neurons using the initial inputs."

	layers first updateWeights: initialInputs
]
