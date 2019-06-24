"
This class represent an artificial neuron for machine learning neural architectures.

I implement this class in the context of Agile Artificial Intelligence book.

In regard of the authors of agileArtificialIntelligence book. But we have added a condition to check if the input is a volume for a convolutional purpose.

Public API and Key Messages

- message one   
- message two 
- (for bonus points) how to create instances.

   One simple example is simply gorgeous.
 
Internal Representation and Key Implementation Points.

    Instance Variables
	bias:		<Number>
	weights:		<Array>


    Implementation Points
"
Class {
	#name : #Neuron,
	#superclass : #Object,
	#instVars : [
		'weights',
		'bias',
		'learningRate',
		'activationFunction',
		'delta',
		'output'
	],
	#category : #NeuralNetwork
}

{ #category : #accessing }
Neuron >> adjustBias [
	"To update bias values on backward propagation algorithm."

	bias := bias + (learningRate * delta)
]

{ #category : #accessing }
Neuron >> adjustDeltaWith: anError [
	"To update delta value on backward propagation algorithm."

	delta := anError * (activationFunction derivative: output)
]

{ #category : #accessing }
Neuron >> adjustWeightsWithInput: inputs [
	"To update weights values on backward propagation algorithm."

	inputs withIndexDo: [ :anInput :index |
		weights at: index put: ((weights at: index) + (learningRate * delta * anInput)) ]
]

{ #category : #accessing }
Neuron >> bias [
	"To get the instance variable bias"

	^ bias
]

{ #category : #accessing }
Neuron >> bias: aNumber [
	"To set the instance variable bias"

	bias := aNumber
]

{ #category : #accessing }
Neuron >> delta [
	"To get the instance variable delta"

	^ delta
]

{ #category : #'as yet unclassified' }
Neuron >> feed: inputs [
	"To apply a set of input values and obtaining the output value of a simple neuron"

	| z |
	z := (inputs with: weights collect: [ :x :w | x * w]) sum + self bias.
	output := activationFunction eval: z.
	^ output
]

{ #category : #accessing }
Neuron >> initialize [
	"To properly initialize a neuron with custom params"

	super initialize.
	learningRate := 0.1.
	self sigmoid
]

{ #category : #accessing }
Neuron >> learningRate [
	"To get the instance variable learningRate"

	^ learningRate
]

{ #category : #accessing }
Neuron >> learningRate: aLearningRateAsFloat [
	"To set the instance variable learningRate. The argument should be a small floating value. For example, 0.1"

	learningRate := aLearningRateAsFloat
]

{ #category : #accessing }
Neuron >> output [
	"To get the instance variable output previous computed when sending feed: message."

	^ output
]

{ #category : #accessing }
Neuron >> sigmoid [
	"To specify that the activation function is the sigmoid function"

	activationFunction := SigmoidAF new
]

{ #category : #accessing }
Neuron >> step [
	"To specify that the activation function is the step function"

	activationFunction := StepAF new
]

{ #category : #accessing }
Neuron >> tanh [
	"To specify that the activation function is the hyperbolic tangent function"

	activationFunction := HyperbolicTangentAF new
]

{ #category : #accessing }
Neuron >> train: inputs desiredOutput: desiredOutput [
	"To adjust weights and bias of a neuron for a training data (inputs, desiredOutput)"

	| diff output delta |
	output := self feed: inputs.
	diff := desiredOutput - output.
	delta := diff * (activationFunction derivative: output).
	
	inputs withIndexDo: [ :anInput :index |
				self weights at: index put: ((self weights at: index) + (self learningRate * delta * anInput )) ].
	self bias: self bias + (self learningRate * delta).
]

{ #category : #accessing }
Neuron >> weights [
	"To get the instance variable weights"

	^ weights
]

{ #category : #accessing }
Neuron >> weights: someWeightsAsNumbers [
	"To set the instance variable weights"

	weights := someWeightsAsNumbers
]
