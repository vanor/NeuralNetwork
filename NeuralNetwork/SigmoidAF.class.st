"
THis class represent the sigmoid activation function.

This will improve the behaviour of a perceptron.

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
	#name : #SigmoidAF,
	#superclass : #ActivationFunction,
	#category : #NeuralNetwork
}

{ #category : #'as yet unclassified' }
SigmoidAF >> derivative: output [
	"To transfert derivative error for the backpropagation algorithm"

	^ output * (1 - output)
]

{ #category : #'as yet unclassified' }
SigmoidAF >> eval: z [
	"To evaluate activation function on sum of bias and scalar product of weights and inputs vectors"

	^ 1 / (1 + z negated exp)
]
