"
This class is a convolution layer.

It will contain features and neurons for doing a convolution.

@yvanGuemkam. 

Public API and Key Messages

- message one   
- message two 
- (for bonus points) how to create instances.

   One simple example is simply gorgeous.
 
Internal Representation and Key Implementation Points.

    Instance Variables
	features:		<Object>
	neurons:		<Object>
	outputVolumeSizes:		<Object>
	receptiveFields:		<Object>
	stride:		<Object>
	zeroPaddingSize:		<Object>


    Implementation Points
"
Class {
	#name : #CNeuronLayer,
	#superclass : #Object,
	#instVars : [
		'features',
		'neurons',
		'receptiveFields',
		'paddedInput',
		'stride',
		'zeroPaddingSize',
		'previousLayer',
		'nextLayer'
	],
	#category : #NeuralNetwork
}

{ #category : #'as yet unclassified' }
CNeuronLayer >> configurePaddingSizeForInputSize: inputSize [
	"To calculate the length of zero padding surface."

	| featureSize outputSize |
	featureSize := features anyOne characteristics size.
	outputSize := neurons size.
	receptiveFields := (1 to: outputSize) collect: [ :notUsed | Array new: outputSize ].
	zeroPaddingSize := (((outputSize - 1) * stride) - inputSize + (featureSize)) / 2
]

{ #category : #accessing }
CNeuronLayer >> createPaddedCuboid: inputCuboid [
	"To do zero padding."

	| inputDepth zeroArray nbOfColumns zeroRow |
	inputDepth := inputCuboid anyOne anyOne size.
	zeroArray := (1 to: inputDepth) collect: [ :e | 0 ].
	paddedInput := inputCuboid collect: [ :elmt |
		| t |
		t := elmt.
		zeroPaddingSize timesRepeat: [ t := t copyWithFirst: zeroArray ];
			timesRepeat: [ t := t copyWith: zeroArray ].
		t ].
	nbOfColumns := paddedInput anyOne size.
	zeroRow := (1 to: nbOfColumns) collect: [ :e | zeroArray ].
	zeroPaddingSize timesRepeat: [ 
		paddedInput := paddedInput copyWith: zeroRow.
		paddedInput := paddedInput copyWithFirst: zeroRow ].
]

{ #category : #'as yet unclassified' }
CNeuronLayer >> createReceptiveFields [
	"To create the receptive fields."

	| paddedSize featureSize receptiveField cuboid |
	featureSize := features anyOne characteristics size.
	paddedSize := paddedInput size.
	
	1 to: (paddedSize - featureSize + 1) by: stride do: [ :firstRowIndex |
		1 to: (paddedSize - featureSize + 1) by: stride do: [ :firstColumnIndex |
			| rowIndex columnIndex |
			rowIndex := 1 + ((firstRowIndex - 1) / stride).
			columnIndex := 1 + ((firstColumnIndex - 1) / stride).
			cuboid := (firstRowIndex to: (featureSize + firstRowIndex - 1)) collect: [ :iRow |
						(firstColumnIndex to: (featureSize + firstColumnIndex - 1)) collect: [ :iCol |
						(paddedInput at: iRow) at: iCol ] ].
			receptiveField := ReceptiveField new entries: cuboid; row: rowIndex; column: columnIndex; yourself.
			(receptiveFields at: rowIndex) at: columnIndex put: receptiveField ] ].
]

{ #category : #accessing }
CNeuronLayer >> features [
	"To get features of the layer"

	^ features.
]

{ #category : #accessing }
CNeuronLayer >> features: anArrayOfFeatures [
	"To set features of the layer"

	features := anArrayOfFeatures.
]

{ #category : #initialization }
CNeuronLayer >> feed: anInputVolume [
	"To feed this layer with some input volume value."

	| inputSize outputVolume |
	inputSize := anInputVolume size.
	self configurePaddingSizeForInputSize: inputSize.
	self createPaddedCuboid: anInputVolume.
	self createReceptiveFields;
			setWeightsIntoNeurons;
			feedWithReceptiveField.
	outputVolume := neurons collect: [ :row | row collect: [ :col | col collect: [ :neuron | neuron output ]]].
	(outputVolume anyOne size = 1) ifTrue: [ nextLayer ifNotNil: [ nextLayer feed: (outputVolume flattened) ] ] 
											ifFalse: [ nextLayer ifNotNil: [ nextLayer feed: outputVolume ] ]
]

{ #category : #accessing }
CNeuronLayer >> feedWithReceptiveField [
	"To give proper input in each neuron in the layer."

	features doWithIndex: [ :feature :index |
		neurons doWithIndex: [ :aRowOfNeurons :rowIndex |
			aRowOfNeurons doWithIndex: [ :anArrayOfNeurons :columnIndex |
				(anArrayOfNeurons at: index) feedConvolutional: (((receptiveFields at: rowIndex) at: columnIndex) entries) withBias: (feature bias) ]]].
]

{ #category : #'as yet unclassified' }
CNeuronLayer >> initialize [
	"To initialize a convolution layer with specific stride."

	super initialize.
	stride := 1.
]

{ #category : #'as yet unclassified' }
CNeuronLayer >> initializeForNbOfFeatures: nbOfFeatures withWidth: width height: height depth: depth andOutputSize: outputSize [
	"To configure the layer with an 3D array of features."

	features := (1 to: nbOfFeatures) collect: [ :index |
		Feature new initializeWidth: width height: height depth: depth; depth: index ].
	self neuronsWidth: outputSize height: outputSize depth: nbOfFeatures.
]

{ #category : #accessing }
CNeuronLayer >> neurons [
	"To get the neurons of the layer"

	^ neurons.
]

{ #category : #'as yet unclassified' }
CNeuronLayer >> neuronsWidth: width height: height depth: depth [
	"To configure the layer with an 3D array of neurons."

	neurons := (1 to: width) collect: [ :notUsed1 |
		(1 to: height) collect: [ :notUsed2 |
		(1 to: depth) collect: [ :notUsed3 | Neuron new bias: 0; tanh ] ] ]
]

{ #category : #accessing }
CNeuronLayer >> nextLayer [
	"To get the nextLayer attribute."

	^ nextLayer.
]

{ #category : #accessing }
CNeuronLayer >> nextLayer: aLayer [
	"To set the nextLayer attribute."

	nextLayer := aLayer
]

{ #category : #accessing }
CNeuronLayer >> numberOfFeatures [
	"To get number of features the layer have."

	^ features size.
]

{ #category : #accessing }
CNeuronLayer >> outputVolumeSize [
	"To get outputVolumeSize of the layer"

	^ receptiveFields size.
]

{ #category : #accessing }
CNeuronLayer >> previousLayer [
	"To get the previousLayer attribute."

	^ previousLayer
]

{ #category : #accessing }
CNeuronLayer >> previousLayer: aLayer [
	"To set the previousLayer attribute."

	previousLayer := aLayer
]

{ #category : #accessing }
CNeuronLayer >> setWeightsIntoNeurons [
	"To dispatch shared weights to all neurons that will do a convolution on a filter."

	features doWithIndex: [ :feature :index |
		neurons do: [ :aRowOfNeurons |
			aRowOfNeurons do: [ :anArrayOfNeurons |
				(anArrayOfNeurons at: index) weights: (feature characteristics) ] ] ].
]

{ #category : #accessing }
CNeuronLayer >> stride [
	"To get stride of the layer"

	^ stride.
]

{ #category : #accessing }
CNeuronLayer >> stride: anInteger [
	"To set stride of the layer"

	stride := anInteger.
]

{ #category : #accessing }
CNeuronLayer >> zeroPaddingSize [
	"To get zeroPaddingSize of the layer"

	^ zeroPaddingSize.
]
