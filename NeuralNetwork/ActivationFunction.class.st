"
This class is an abstraction of activation function in NeuralNetwork.

This class should be more defined because it is an abstract class.

In regard of authors of the book AgileArtificialIntelligence. 

Public API and Key Messages

- message one   
- message two 
- (for bonus points) how to create instances.

   One simple example is simply gorgeous.
 
Internal Representation and Key Implementation Points.


    Implementation Points
"
Class {
	#name : #ActivationFunction,
	#superclass : #Object,
	#category : #NeuralNetwork
}

{ #category : #'as yet unclassified' }
ActivationFunction >> derivative: output [
	"To transfert derivative error for the backpropagation algorithm"

	^ self subclassResponsibility
]

{ #category : #'as yet unclassified' }
ActivationFunction >> eval: z [
	"To evaluate activation function on sum of bias and scalar product of weights and inputs vectors"

	^ self subclassResponsibility
]
