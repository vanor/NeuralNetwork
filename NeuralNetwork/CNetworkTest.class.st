"
For testing the CNNetwork class.

This class allow a user to test feeding, backpropagation and others.

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
	#name : #CNetworkTest,
	#superclass : #TestCase,
	#category : #NeuralNetwork
}

{ #category : #tests }
CNetworkTest >> testBasic [
	"To test the feeding of a CNN with four layers. CONV - POOL - CONV - FULLY."

	| r cube cl1 cl2 pl nl net f1 f2 f3 f4 f5 res1 res2 res3 res4 |
	r := Random seed: 42.
	cube := #(
		#(#(2 1 2) #(0 1 2) #(1 1 2) #(2 2 0) #(0 2 1) #(1 0 2))
		#(#(0 1 2) #(2 1 1) #(2 1 0) #(1 2 0) #(0 2 2) #(2 1 0))
		#(#(2 1 1) #(0 0 1) #(0 2 2) #(0 1 1) #(1 0 0) #(0 2 1))
		#(#(1 2 0) #(1 1 1) #(1 1 1) #(1 0 2) #(1 1 2) #(1 0 2))
		#(#(1 0 1) #(2 0 0) #(1 0 2) #(1 0 0) #(2 2 0) #(2 1 0))
		#(#(1 0 2) #(1 0 0) #(1 0 1) #(2 0 2) #(1 1 0) #(0 2 1))).
		
	cl1 := CNeuronLayer new initializeForNbOfFeatures: 2 withWidth: 2 height: 2 depth: 3 andOutputSize: 4.
	cl1 stride: 2.
	f1 := Feature new bias: 1; characteristics: #(
		#(#(1 0 1) #(0 0 1))
		#(#(1 1 0) #(1 0 0))).
	f2 := Feature new bias: 0; characteristics: #(
		#(#(1 1 0) #(1 0 0))
		#(#(1 0 1) #(0 0 1))).
	cl1 features: { f1 . f2 }.

	pl := PoolingLayer new initializeForReceptiveFieldSize: 2 withStride: 2.

	cl2 := CNeuronLayer new initializeForNbOfFeatures: 3 withWidth: 2 height: 2 depth: 2 andOutputSize: 1.
	cl2 stride: 2.
	f3 := Feature new bias: 0; characteristics: #(
		#(#(1 1) #(1 0))
		#(#(0 1) #(0 0))).
	f4 := Feature new bias: 1; characteristics: #(
		#(#(0 0) #(1 0))
		#(#(0 1) #(1 1))).
	f5 := Feature new bias: 0; characteristics: #(
		#(#(0 1) #(1 1))
		#(#(0 0) #(1 0))).
	cl2 features: { f3 . f4 . f5 }.

	nl := NeuronLayer new.
	nl initializeNbOfNeurons: 3 nbOfWeights: 3 using: r.
	nl neurons collectWithIndex: [ :neuron :index | |weights|
		weights := (1 to: 3) collect: [ :ii | index + ii ].
		neuron weights: weights; bias: index ].

	net := CNNetwork new.
	net addLayer: cl1; addLayer: pl; addLayer: cl2; addLayer: nl.
	net feed: cube.
	
	res1 := cl1 neurons collect: [ :row | row collect: [ :col | col collect: #output ] ].
	res2 := pl units collect: [ :row | row collect: #outputArray ].
	res3 := cl2 neurons collect: [ :row | row collect: [ :col | col collect: #output ] ].
	res4 := nl neurons collect: #output.
	
	self assert: (((res1 at: 1) at: 1) at: 1) equals: 0.9950547536867305. "tanh 3"
	self assert: (((res1 at: 1) at: 2) at: 1) equals: 0.9950547536867305. "tanh 3"
	self assert: (((res1 at: 1) at: 3) at: 1) equals: 0.9999092042625951. "tanh 5"
	self assert: (((res1 at: 1) at: 4) at: 1) equals: 0.9640275800758169. "tanh 2"
	self assert: (((res1 at: 2) at: 1) at: 1) equals: 0.9999092042625951. "tanh 5"
	self assert: (((res1 at: 2) at: 2) at: 1) equals: 0.999329299739067. "tanh 4"
	self assert: (((res1 at: 2) at: 3) at: 1) equals: 0.9999877116507956. "tanh 6"
	self assert: (((res1 at: 2) at: 4) at: 1) equals: 0.9999092042625951. "tanh 5"
	self assert: (((res1 at: 3) at: 1) at: 1) equals: 0.9640275800758169. "tanh 2"
	self assert: (((res1 at: 3) at: 2) at: 1) equals: 0.9999983369439447. "tanh 7"
	self assert: (((res1 at: 3) at: 3) at: 1) equals: 0.999999969540041. "tanh 9"
	self assert: (((res1 at: 3) at: 4) at: 1) equals: 0.9999983369439447. "tanh 7"
	self assert: (((res1 at: 4) at: 1) at: 1) equals: 0.9950547536867305. "tanh 3"
	self assert: (((res1 at: 4) at: 2) at: 1) equals: 0.9950547536867305. "tanh 3"
	self assert: (((res1 at: 4) at: 3) at: 1) equals: 0.9999092042625951. "tanh 5"
	self assert: (((res1 at: 4) at: 4) at: 1) equals: 0.9640275800758169. "tanh 2"
	
	self assert: (((res1 at: 1) at: 1) at: 2) equals: 0.9640275800758169. "tanh 2"
	self assert: (((res1 at: 1) at: 2) at: 2) equals: 0.999329299739067. "tanh 4"
	self assert: (((res1 at: 1) at: 3) at: 2) equals: 0.9950547536867305. "tanh 3"
	self assert: (((res1 at: 1) at: 4) at: 2) equals: 0.9950547536867305. "tanh 3"
	self assert: (((res1 at: 2) at: 1) at: 2) equals: 0.7615941559557649. "tanh 1"
	self assert: (((res1 at: 2) at: 2) at: 2) equals: 0.9999997749296758. "tanh 8"
	self assert: (((res1 at: 2) at: 3) at: 2) equals: 0.999329299739067. "tanh 4"
	self assert: (((res1 at: 2) at: 4) at: 2) equals: 0.999329299739067. "tanh 4"
	self assert: (((res1 at: 3) at: 1) at: 2) equals: 0.9640275800758169. "tanh 2"
	self assert: (((res1 at: 3) at: 2) at: 2) equals: 0.9999983369439447. "tanh 7"
	self assert: (((res1 at: 3) at: 3) at: 2) equals: 0.9950547536867305. "tanh 3"
	self assert: (((res1 at: 3) at: 4) at: 2) equals: 0.9950547536867305. "tanh 3"
	self assert: (((res1 at: 4) at: 1) at: 2) equals: 0.7615941559557649. "tanh 1"
	self assert: (((res1 at: 4) at: 2) at: 2) equals: 0.9640275800758169. "tanh 2"
	self assert: (((res1 at: 4) at: 3) at: 2) equals: 0.9950547536867305. "tanh 3"
	self assert: (((res1 at: 4) at: 4) at: 2) equals: 0.9640275800758169. "tanh 2"
	
	self assert: (((res2 at: 1) at: 1) at: 1) equals: 0.9999092042625951. "tanh 5"
	self assert: (((res2 at: 1) at: 2) at: 1) equals: 0.9999877116507956. "tanh 6"
	self assert: (((res2 at: 2) at: 1) at: 1) equals: 0.9999983369439447. "tanh 7"
	self assert: (((res2 at: 2) at: 2) at: 1) equals: 0.999999969540041. "tanh 9"
	
	self assert: (((res2 at: 1) at: 1) at: 2) equals: 0.9999997749296758. "tanh 8"
	self assert: (((res2 at: 1) at: 2) at: 2) equals: 0.999329299739067. "tanh 4"
	self assert: (((res2 at: 2) at: 1) at: 2) equals: 0.9999983369439447. "tanh 7"
	self assert: (((res2 at: 2) at: 2) at: 2) equals: 0.9950547536867305. "tanh 3"
	
	self assert: (((res3 at: 1) at: 1) at: 1) equals: 0.999329158961739. "tanh (tanh 5 + tanh 6 +tanh 7 +tanh 8) == x1"
	self assert: (((res3 at: 1) at: 1) at: 2) equals: 0.9999082992696797. "tanh (1 + tanh 3 + tanh 6 +tanh 7 +tanh 9) == x2"
	self assert: (((res3 at: 1) at: 1) at: 3) equals: 0.9993283829165195. "tanh (tanh 4 + tanh 6 +tanh 8 +tanh 9) == x3"
	
	self assert: (res4 at: 1) equals: 0.9999544063607851. "sigmoid ((2*x1) + (3*x2) + (4*x3) +1)"
	self assert: (res4 at: 2) equals: 0.9999991636874457. "sigmoid ((3*x1) + (4*x2) + (5*x3) +2)"
	self assert: (res4 at: 3) equals: 0.9999999846604051. "sigmoid ((4*x1) + (5*x2) + (6*x3) +3)"
	self assert: (net outputLayer neurons collect: #output) equals: res4.
]
