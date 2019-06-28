"
This class implement a pooling layer.

This class allow to do a pooling operation in Convolutional Networks.

@yvanGuemkam

Public API and Key Messages

- message one   
- message two 
- (for bonus points) how to create instances.

   One simple example is simply gorgeous.
 
Internal Representation and Key Implementation Points.

    Instance Variables
	mode:		<Object>
	receptiveFields:		<Object>
	stride:		<Object>
	units:		<Object>


    Implementation Points
"
Class {
	#name : #PoolingLayer,
	#superclass : #Object,
	#instVars : [
		'mode',
		'units',
		'receptiveFields',
		'stride',
		'receptiveFieldSize',
		'previousLayer',
		'nextLayer'
	],
	#category : #NeuralNetwork
}

{ #category : #initialization }
PoolingLayer >> configureOutputForInputSize: inputSize withStride: aStride andReceptiveFieldSize: anInteger [
	"To compute the output volume size and make some configurations."

	| outputSize |
	receptiveFieldSize := anInteger.
	self stride: aStride.
	outputSize := ((inputSize - receptiveFieldSize) / stride) + 1.
	units := (1 to: outputSize) collect: [ :notUsed1 | 
		(1 to: outputSize) collect: [ :notUsed2 | PoolingUnit new ] ].
	receptiveFields := (1 to: outputSize) collect: [ :notUsed | Array new: outputSize ].
]

{ #category : #initialization }
PoolingLayer >> createReceptiveFields: inputs [
	"To create receptive fields for doing the pooling."

	| inputSize cuboid receptiveField |
	inputSize := inputs size.
	
	1 to: (inputSize - receptiveFieldSize + 1) by: stride do: [ :firstRowIndex |
		1 to: (inputSize - receptiveFieldSize + 1) by: stride do: [ :firstColumnIndex |
			| rowIndex columnIndex |
			rowIndex := 1 + ((firstRowIndex - 1) / stride).
			columnIndex := 1 + ((firstColumnIndex - 1) / stride).
			cuboid := (firstRowIndex to: (receptiveFieldSize + firstRowIndex - 1)) collect: [ :iRow |
						(firstColumnIndex to: (receptiveFieldSize + firstColumnIndex - 1)) collect: [ :iCol |
						(inputs at: iRow) at: iCol ] ].
			receptiveField := ReceptiveField new entries: cuboid; row: rowIndex; column: columnIndex; yourself.
			(receptiveFields at: rowIndex) at: columnIndex put: receptiveField ] ].
]

{ #category : #initialization }
PoolingLayer >> feed: inputVolume [
	"To execute the pooling on some inputs."

	| inputSize outputVolume |
	inputSize := inputVolume size.
	self configureOutputForInputSize: inputSize withStride: stride andReceptiveFieldSize: 3.
	self createReceptiveFields: inputVolume.
	self feedWithReceptiveField.
	outputVolume := units collect: [ :row | row collect: [ :unit | unit outputArray ]].
	nextLayer ifNotNil: [ nextLayer feed: outputVolume ].
]

{ #category : #initialization }
PoolingLayer >> feedWithReceptiveField [
	"To evaluate the result of the pooling layer on an input volume."

	units doWithIndex: [ :columnUnits :rowIndex |
		columnUnits doWithIndex: [ :unit :columnIndex | unit feed: (((receptiveFields at: rowIndex) at: columnIndex) entries)]]
]

{ #category : #initialization }
PoolingLayer >> initialize [
	"To custom initialization of PoolingLayer to be a max pooling by default."

	super initialize.
	mode := 'MAX'.
]

{ #category : #accessing }
PoolingLayer >> mode [
	"To get mode attribute of the layer."

	^ mode
]

{ #category : #accessing }
PoolingLayer >> mode: aString [
	"To set mode attribute of the layer. This mode should be MAX or AVG."

	mode := aString
]

{ #category : #accessing }
PoolingLayer >> nextLayer [
	"To get the nextLayer attribute."

	^ nextLayer
]

{ #category : #accessing }
PoolingLayer >> nextLayer: aLayer [
	"To set the nextLayer attribute."

	nextLayer := aLayer
]

{ #category : #accessing }
PoolingLayer >> previousLayer [
	"To get the previousLayer attribute."

	^ previousLayer
]

{ #category : #accessing }
PoolingLayer >> previousLayer: aLayer [
	"To set the previousLayer attribute."

	previousLayer := aLayer
]

{ #category : #accessing }
PoolingLayer >> stride [
	"To get stride attribute of the layer."

	^ stride
]

{ #category : #accessing }
PoolingLayer >> stride: anInteger [
	"To set stride attribute of the layer."

	stride := anInteger
]

{ #category : #accessing }
PoolingLayer >> units [
	"To get units attribute of the layer."

	^ units
]
