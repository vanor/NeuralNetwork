"
This class represent Convolutional neural networks.

This class allow to create, configure and manage this kind of network.

@yvanGuemkam

Public API and Key Messages

- message one   
- message two 
- (for bonus points) how to create instances.

   One simple example is simply gorgeous.
 
Internal Representation and Key Implementation Points.


    Implementation Points
"
Class {
	#name : #CNNetwork,
	#superclass : #Object,
	#instVars : [
		'layers',
		'errors',
		'precisions'
	],
	#category : #NeuralNetwork
}

{ #category : #adding }
CNNetwork >> addLayer: aLayer [
	"To add a layer to a convolutional neural network."
	
	layers ifNotEmpty: [ 
		aLayer previousLayer: layers last.
		layers last nextLayer: aLayer ].
	layers add: aLayer.
]

{ #category : #initialization }
CNNetwork >> backwardPropagateError: expectedOutputs [
	"To propagate the error on expectedOutputs corresponding to the desired outputs on a trainig sample."

	self outputLayer backwardPropagateError: expectedOutputs
]

{ #category : #initialization }
CNNetwork >> feed: anInputVolume [
	"To feed the network with a specific input value."

	^ layers first feed: anInputVolume
]

{ #category : #initialization }
CNNetwork >> initialize [
	"To properly initialize the network."

	layers := OrderedCollection new.
	errors := OrderedCollection new.
	precisions := OrderedCollection new.
]

{ #category : #initialization }
CNNetwork >> learningRate: aLearningRate [
	"To set the learning rate of the whole networkk with a specific value."

	layers do: [ :layer |
		layer class = PoolingLayer ifFalse: [ layer learningRate: aLearningRate ] ]
]

{ #category : #initialization }
CNNetwork >> numberOfOutputs [
	"To get the number of output of the network."

	^ layers last numberOfNeurons
]

{ #category : #initialization }
CNNetwork >> outputLayer [
	"To get the output layer of the network."

	^ layers last
]

{ #category : #'as yet unclassified' }
CNNetwork >> outputWidth [
	"To get the output width of a the first convolutional layer of the network."

	self assert: [ layers first neurons notEmpty ] description: 'outputWidth should be invoked after configuring the layer'.
	^ layers first neurons size.
]

{ #category : #running }
CNNetwork >> predict: inputs [
	"To make a prediction. This method assumes that the number of outputs is the same than the number of different values the network can output."

	| outputs |
	outputs := self feed: inputs.
	^ (outputs indexOf: (outputs max)) - 1
]

{ #category : #initialization }
CNNetwork >> train: trainingSet withLabels: labelSet andNbEpoch: nbOfEpoch [
	"To train the network using the training data set."

	| sumError outputs expectedOutputs epochPrecision t |
	1 to: nbOfEpoch do: [ :epoch |
		sumError := 0.
		epochPrecision := 0.
		trainingSet with: labelSet do: [ :data :label |
			outputs := self feed: data.
			expectedOutputs := (1 to: self numberOfOutputs) collect: [ :notUsed | 0 ].
			expectedOutputs at: label + 1 put: 1.
			(label = (self predict: data)) ifTrue: [ epochPrecision := epochPrecision + 1 ].
			t := (1 to: expectedOutputs size)
						collect: [ :i | ((expectedOutputs at: i) - (outputs at: i)) raisedTo: 2 ].
			sumError := sumError + t sum.
			
			self backwardPropagateError: expectedOutputs.
			self updateWeights.
		].
		errors add: sumError.
		precisions add: (epochPrecision / trainingSet size) asFloat.
	]
]

{ #category : #initialization }
CNNetwork >> updateWeights [
	"To update the weights of the neurons using the padded inputs."

	layers first updateWeights
]
