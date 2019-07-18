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

{ #category : #'as yet unclassified' }
CNetworkTest >> constructCuboid [
	"To construct data for feeding the network."

	| cube |
	cube := #(
		#(#(2 1 2) #(0 1 2) #(1 1 2) #(2 2 0) #(0 2 1) #(1 0 2))
		#(#(0 1 2) #(2 1 1) #(2 1 0) #(1 2 0) #(0 2 2) #(2 1 0))
		#(#(2 1 1) #(0 0 1) #(0 2 2) #(0 1 1) #(1 0 0) #(0 2 1))
		#(#(1 2 0) #(1 1 1) #(1 1 1) #(1 0 2) #(1 1 2) #(1 0 2))
		#(#(1 0 1) #(2 0 0) #(1 0 2) #(1 0 0) #(2 2 0) #(2 1 0))
		#(#(1 0 2) #(1 0 0) #(1 0 1) #(2 0 2) #(1 1 0) #(0 2 1))).
		
	^ cube
]

{ #category : #'as yet unclassified' }
CNetworkTest >> constructNetwork [
	"To construct a simple CNN with four layers. CONV - POOL - CONV - FULLY."

	| cl1 cl2 pl nl net f1 f2 f3 f4 f5 |
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
	nl initializeNbOfNeurons: 3 nbOfWeights: 3 using: (Random seed: 42).
	nl neurons collectWithIndex: [ :neuron :index | |weights|
		weights := (1 to: 3) collect: [ :ii | index + ii ].
		neuron weights: weights; bias: index ].

	net := CNNetwork new.
	net addLayer: cl1; addLayer: pl; addLayer: cl2; addLayer: nl.
	
	^ net
]

{ #category : #tests }
CNetworkTest >> testBeforeLastLayerComputationOfDelta [
	"To test if the layer just before the last layer well evaluate neurons delta."

	| net cube result |
	net := self constructNetwork.
	cube := self constructCuboid.
	net train: { cube } withLabels: { 2 } andNbEpoch: 1.
	result := net layers third neurons collect: [ :row | row collect: [ :col | col collect: #delta ] ].
	
	self assert: (result anyOne anyOne size) equals: 3. "delta next layer == d and output this layer == s"
	self assert: (result anyOne anyOne at: 1) equals: -0.0001872501367200737. "(d1*2 + d2*3 + d3*4) * f'(s1)"
	self assert: (result anyOne anyOne at: 2) equals: -0.00028020168386293416. "(d1*3 + d2*4 + d3*5) * f'(s1)"
	self assert: (result anyOne anyOne at: 3) equals: -0.00037282848348563707. "(d1*4 + d2*5 + d3*6) * f'(s1)"
]

{ #category : #tests }
CNetworkTest >> testFeed [
	"To test the feeding of a CNN with four layers. CONV - POOL - CONV - FULLY."

	| cube net res1 res2 res3 res4 |
	cube := self constructCuboid.
	net := self constructNetwork.
	net feed: cube.
	
	res1 := (net layers first) neurons collect: [ :row | row collect: [ :col | col collect: #output ] ].
	res2 := (net layers second) units collect: [ :row | row collect: #outputArray ].
	res3 := (net layers third) neurons collect: [ :row | row collect: [ :col | col collect: #output ] ].
	res4 := (net layers fourth) neurons collect: #output.
	
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

{ #category : #tests }
CNetworkTest >> testLastLayerComputationOfDelta [
	"To test if the last layer well evaluate neurons delta."

	| net cube result |
	net := self constructNetwork.
	cube := self constructCuboid.
	net train: { cube } withLabels: { 2 } andNbEpoch: 1.
	result := net layers last neurons collect: #delta.
	
	self assert: (result size) equals: 3.
	self assert: (result at: 1) equals: -4.558948174976358e-5. "-0.9999544063607851 * f'(0.9999544063607851)"
	self assert: (result at: 2) equals: -8.363111554830259e-7. "-0.9999991636874457* f'(0.9999991636874457)"
	self assert: (result at: 3) equals: 2.353031689337116e-16. "1.5339594927610278e-8 * f'(0.9999999846604051)"
]

{ #category : #tests }
CNetworkTest >> testSecondLayerComputationOfDelta [
	"To test if this layer compute delta properly."

	| net cube result |
	net := self constructNetwork.
	cube := self constructCuboid.
	net train: { cube } withLabels: { 2 } andNbEpoch: 1.
	result := net layers second deltas.
	
	self assert: (result size) equals: 2.
	self assert: (result anyOne size) equals: 2.
	self assert: (result anyOne anyOne size) equals: 2.
	
	self assert: (result first first first) equals: 1. "d1,1,1 * w1,1,1,1 + d2,1,1 * w2,1,1,1 + d3,1,1 * w3,1,1,1"
]
