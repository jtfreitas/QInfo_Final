Øê
×
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
e
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2		
@
ReadVariableOp
resource
value"dtype"
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Ö
d
mps1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namemps1
]
mps1/Read/ReadVariableOpReadVariableOpmps1*
_output_shapes

:*
dtype0
h
mps2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemps2
a
mps2/Read/ReadVariableOpReadVariableOpmps2*"
_output_shapes
:*
dtype0
`
biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namebias
Y
bias/Read/ReadVariableOpReadVariableOpbias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
t
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0

NoOpNoOp
¶
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ñ
valueçBä BÝ

layer-0
layer_with_weights-0
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures*
* 
®
mps1
mps2
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
:
iter
	decay
learning_rate
momentum*

0
1
2*

0
1
2*
* 
°
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
* 
* 
* 

serving_default* 
RL
VARIABLE_VALUEmps14layer_with_weights-0/mps1/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEmps24layer_with_weights-0/mps2/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEbias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1
2*

0
1
2*
* 

non_trainable_variables

 layers
!metrics
"layer_regularization_losses
#layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
KE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

$0
%1*
* 
* 
* 
* 
* 
* 
* 
* 
8
	&total
	'count
(	variables
)	keras_api*
[
*
thresholds
+true_positives
,false_positives
-	variables
.	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

&0
'1*

(	variables*
* 
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

+0
,1*

-	variables*

serving_default_input_1Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
Ì
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1mps1mps2bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_38067
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
û
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemps1/Read/ReadVariableOpmps2/Read/ReadVariableOpbias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_38332

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemps1mps2biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcounttrue_positivesfalse_positives*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_38375éé

Þ
ä
@__inference_model_layer_call_and_return_conditional_losses_37541

inputs
tn__toy_37533:#
tn__toy_37535:
tn__toy_37537:
identity¢tn__toy/StatefulPartitionedCallý
tn__toy/StatefulPartitionedCallStatefulPartitionedCallinputstn__toy_37533tn__toy_37535tn__toy_37537*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_tn__toy_layer_call_and_return_conditional_losses_37532w
IdentityIdentity(tn__toy/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
NoOpNoOp ^tn__toy/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2B
tn__toy/StatefulPartitionedCalltn__toy/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
Ý
B__inference_tn__toy_layer_call_and_return_conditional_losses_37532

inputs3
!loop_body_readvariableop_resource:9
#loop_body_readvariableop_1_resource:3
%loop_body_add_readvariableop_resource:
identity¢loop_body/ReadVariableOp¢loop_body/ReadVariableOp_1¢loop_body/add/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
Rank/packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:W
	Max/inputPackstrided_slice:output:0*
N*
T0*
_output_shapes
:O
MaxMaxMax/input:output:0range:output:0*
T0*
_output_shapes
: h
&loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : 
 loop_body/PlaceholderWithDefaultPlaceholderWithDefault/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: E
loop_body/ShapeShapeinputs*
T0*
_output_shapes
:g
loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
loop_body/strided_sliceStridedSliceloop_body/Shape:output:0&loop_body/strided_slice/stack:output:0(loop_body/strided_slice/stack_1:output:0(loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :}
loop_body/GreaterGreater loop_body/strided_slice:output:0loop_body/Greater/y:output:0*
T0*
_output_shapes
: V
loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B :  
loop_body/SelectV2SelectV2loop_body/Greater:z:0)loop_body/PlaceholderWithDefault:output:0loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: Y
loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ±
loop_body/GatherV2GatherV2inputsloop_body/SelectV2:output:0 loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes

:z
loop_body/ReadVariableOpReadVariableOp!loop_body_readvariableop_resource*
_output_shapes

:*
dtype0
loop_body/ReadVariableOp_1ReadVariableOp#loop_body_readvariableop_1_resource*"
_output_shapes
:*
dtype0s
"loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       
loop_body/Tensordot/transpose	Transposeloop_body/GatherV2:output:0+loop_body/Tensordot/transpose/perm:output:0*
T0*
_output_shapes

:
loop_body/Tensordot/MatMulMatMul!loop_body/Tensordot/transpose:y:0 loop_body/ReadVariableOp:value:0*
T0*
_output_shapes

:y
$loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¬
loop_body/Tensordot_1/transpose	Transpose"loop_body/ReadVariableOp_1:value:0-loop_body/Tensordot_1/transpose/perm:output:0*
T0*"
_output_shapes
:t
#loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ¤
loop_body/Tensordot_1/ReshapeReshape#loop_body/Tensordot_1/transpose:y:0,loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:v
%loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ©
loop_body/Tensordot_1/Reshape_1Reshape$loop_body/Tensordot/MatMul:product:0.loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:¡
loop_body/Tensordot_1/MatMulMatMul&loop_body/Tensordot_1/Reshape:output:0(loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes

:e
loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
loop_body/Tensordot_1Reshape&loop_body/Tensordot_1/MatMul:product:0$loop_body/Tensordot_1/shape:output:0*
T0*
_output_shapes
:~
loop_body/add/ReadVariableOpReadVariableOp%loop_body_add_readvariableop_resource*
_output_shapes
:*
dtype0
loop_body/addAddV2loop_body/Tensordot_1:output:0$loop_body/add/ReadVariableOp:value:0*
T0*
_output_shapes
:\
pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:g
pfor/ReshapeReshapeMax:output:0pfor/Reshape/shape:output:0*
T0*
_output_shapes
:R
pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : R
pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :|

pfor/rangeRangepfor/range/start:output:0Max:output:0pfor/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : _
loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :
loop_body/SelectV2/pfor/addAddV2%loop_body/SelectV2/pfor/Rank:output:0&loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: `
loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :`
loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : a
loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
loop_body/SelectV2/pfor/add_1AddV2'loop_body/SelectV2/pfor/Rank_2:output:0(loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: 
loop_body/SelectV2/pfor/MaximumMaximum'loop_body/SelectV2/pfor/Rank_1:output:0loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: 
!loop_body/SelectV2/pfor/Maximum_1Maximum!loop_body/SelectV2/pfor/add_1:z:0#loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: `
loop_body/SelectV2/pfor/ShapeShapepfor/range:output:0*
T0*
_output_shapes
:
loop_body/SelectV2/pfor/subSub%loop_body/SelectV2/pfor/Maximum_1:z:0'loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: o
%loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: 
loop_body/SelectV2/pfor/ReshapeReshapeloop_body/SelectV2/pfor/sub:z:0.loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:l
"loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:
loop_body/SelectV2/pfor/TileTile+loop_body/SelectV2/pfor/Tile/input:output:0(loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: u
+loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
%loop_body/SelectV2/pfor/strided_sliceStridedSlice&loop_body/SelectV2/pfor/Shape:output:04loop_body/SelectV2/pfor/strided_slice/stack:output:06loop_body/SelectV2/pfor/strided_slice/stack_1:output:06loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskw
-loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:y
/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: y
/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ë
'loop_body/SelectV2/pfor/strided_slice_1StridedSlice&loop_body/SelectV2/pfor/Shape:output:06loop_body/SelectV2/pfor/strided_slice_1/stack:output:08loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:08loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maske
#loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
loop_body/SelectV2/pfor/concatConcatV2.loop_body/SelectV2/pfor/strided_slice:output:0%loop_body/SelectV2/pfor/Tile:output:00loop_body/SelectV2/pfor/strided_slice_1:output:0,loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
!loop_body/SelectV2/pfor/Reshape_1Reshapepfor/range:output:0'loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
 loop_body/SelectV2/pfor/SelectV2SelectV2loop_body/Greater:z:0*loop_body/SelectV2/pfor/Reshape_1:output:0loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
%loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : è
 loop_body/GatherV2/pfor/GatherV2GatherV2inputs)loop_body/SelectV2/pfor/SelectV2:output:0.loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
(loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :´
&loop_body/Tensordot/transpose/pfor/addAddV2+loop_body/Tensordot/transpose/perm:output:01loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:|
2loop_body/Tensordot/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: p
.loop_body/Tensordot/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
)loop_body/Tensordot/transpose/pfor/concatConcatV2;loop_body/Tensordot/transpose/pfor/concat/values_0:output:0*loop_body/Tensordot/transpose/pfor/add:z:07loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Î
,loop_body/Tensordot/transpose/pfor/Transpose	Transpose)loop_body/GatherV2/pfor/GatherV2:output:02loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%loop_body/Tensordot/MatMul/pfor/ShapeShape0loop_body/Tensordot/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:q
/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ú
%loop_body/Tensordot/MatMul/pfor/splitSplit8loop_body/Tensordot/MatMul/pfor/split/split_dim:output:0.loop_body/Tensordot/MatMul/pfor/Shape:output:0*
T0*&
_output_shapes
:::*
	num_splitp
-loop_body/Tensordot/MatMul/pfor/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB r
/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB ½
'loop_body/Tensordot/MatMul/pfor/ReshapeReshape.loop_body/Tensordot/MatMul/pfor/split:output:08loop_body/Tensordot/MatMul/pfor/Reshape/shape_1:output:0*
T0*
_output_shapes
: r
/loop_body/Tensordot/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB t
1loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB Á
)loop_body/Tensordot/MatMul/pfor/Reshape_1Reshape.loop_body/Tensordot/MatMul/pfor/split:output:1:loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: r
/loop_body/Tensordot/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB t
1loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB Á
)loop_body/Tensordot/MatMul/pfor/Reshape_2Reshape.loop_body/Tensordot/MatMul/pfor/split:output:2:loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ±
#loop_body/Tensordot/MatMul/pfor/mulMul0loop_body/Tensordot/MatMul/pfor/Reshape:output:02loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: Â
/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack'loop_body/Tensordot/MatMul/pfor/mul:z:02loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:Û
)loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape0loop_body/Tensordot/transpose/pfor/Transpose:y:08loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸
&loop_body/Tensordot/MatMul/pfor/MatMulMatMul2loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0 loop_body/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
1loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePack0loop_body/Tensordot/MatMul/pfor/Reshape:output:02loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0:loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:Ö
)loop_body/Tensordot/MatMul/pfor/Reshape_4Reshape0loop_body/Tensordot/MatMul/pfor/MatMul:product:08loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
+loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2pfor/Reshape:output:0.loop_body/Tensordot_1/Reshape_1/shape:output:09loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:×
,loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape2loop_body/Tensordot/MatMul/pfor/Reshape_4:output:04loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'loop_body/Tensordot_1/MatMul/pfor/ShapeShape5loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
5loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:û
/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0>loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskË
)loop_body/Tensordot_1/MatMul/pfor/MinimumMinimum8loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: k
)loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :´
'loop_body/Tensordot_1/MatMul/pfor/EqualEqual-loop_body/Tensordot_1/MatMul/pfor/Minimum:z:02loop_body/Tensordot_1/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: 
*loop_body/Tensordot_1/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          
*loop_body/Tensordot_1/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          î
(loop_body/Tensordot_1/MatMul/pfor/SelectSelect+loop_body/Tensordot_1/MatMul/pfor/Equal:z:03loop_body/Tensordot_1/MatMul/pfor/Select/t:output:03loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
'loop_body/Tensordot_1/MatMul/pfor/stackPack:loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:ê
+loop_body/Tensordot_1/MatMul/pfor/transpose	Transpose5loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:01loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÍ
)loop_body/Tensordot_1/MatMul/pfor/ReshapeReshape/loop_body/Tensordot_1/MatMul/pfor/transpose:y:00loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)loop_body/Tensordot_1/MatMul/pfor/Shape_1Shape2loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:s
1loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : â
'loop_body/Tensordot_1/MatMul/pfor/splitSplit:loop_body/Tensordot_1/MatMul/pfor/split/split_dim:output:02loop_body/Tensordot_1/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_splitt
1loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB v
3loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB Ç
+loop_body/Tensordot_1/MatMul/pfor/Reshape_1Reshape0loop_body/Tensordot_1/MatMul/pfor/split:output:0<loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: t
1loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB v
3loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB Ç
+loop_body/Tensordot_1/MatMul/pfor/Reshape_2Reshape0loop_body/Tensordot_1/MatMul/pfor/split:output:1<loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: t
1loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB v
3loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB Ç
+loop_body/Tensordot_1/MatMul/pfor/Reshape_3Reshape0loop_body/Tensordot_1/MatMul/pfor/split:output:2<loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ¹
%loop_body/Tensordot_1/MatMul/pfor/mulMul4loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: È
1loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePack4loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:0)loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:á
+loop_body/Tensordot_1/MatMul/pfor/Reshape_4Reshape2loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0:loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÂ
(loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul&loop_body/Tensordot_1/Reshape:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
3loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
1loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePack<loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:î
+loop_body/Tensordot_1/MatMul/pfor/Reshape_5Reshape2loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0:loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ã
-loop_body/Tensordot_1/MatMul/pfor/transpose_1	Transpose4loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0;loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
&loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : É
!loop_body/Tensordot_1/pfor/concatConcatV2pfor/Reshape:output:0$loop_body/Tensordot_1/shape:output:0/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:¾
"loop_body/Tensordot_1/pfor/ReshapeReshape1loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:0*loop_body/Tensordot_1/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
loop_body/add/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :[
loop_body/add/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :Z
loop_body/add/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :
loop_body/add/pfor/addAddV2"loop_body/add/pfor/Rank_1:output:0!loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: 
loop_body/add/pfor/MaximumMaximumloop_body/add/pfor/add:z:0 loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: s
loop_body/add/pfor/ShapeShape+loop_body/Tensordot_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
loop_body/add/pfor/subSubloop_body/add/pfor/Maximum:z:0 loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: j
 loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
loop_body/add/pfor/ReshapeReshapeloop_body/add/pfor/sub:z:0)loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:g
loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:
loop_body/add/pfor/TileTile&loop_body/add/pfor/Tile/input:output:0#loop_body/add/pfor/Reshape:output:0*
T0*
_output_shapes
: p
&loop_body/add/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(loop_body/add/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(loop_body/add/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
 loop_body/add/pfor/strided_sliceStridedSlice!loop_body/add/pfor/Shape:output:0/loop_body/add/pfor/strided_slice/stack:output:01loop_body/add/pfor/strided_slice/stack_1:output:01loop_body/add/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskr
(loop_body/add/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*loop_body/add/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: t
*loop_body/add/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:´
"loop_body/add/pfor/strided_slice_1StridedSlice!loop_body/add/pfor/Shape:output:01loop_body/add/pfor/strided_slice_1/stack:output:03loop_body/add/pfor/strided_slice_1/stack_1:output:03loop_body/add/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask`
loop_body/add/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ö
loop_body/add/pfor/concatConcatV2)loop_body/add/pfor/strided_slice:output:0 loop_body/add/pfor/Tile:output:0+loop_body/add/pfor/strided_slice_1:output:0'loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ª
loop_body/add/pfor/Reshape_1Reshape+loop_body/Tensordot_1/pfor/Reshape:output:0"loop_body/add/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
loop_body/add/pfor/AddV2AddV2%loop_body/add/pfor/Reshape_1:output:0$loop_body/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
SoftmaxSoftmaxloop_body/add/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^loop_body/ReadVariableOp^loop_body/ReadVariableOp_1^loop_body/add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 24
loop_body/ReadVariableOploop_body/ReadVariableOp28
loop_body/ReadVariableOp_1loop_body/ReadVariableOp_12<
loop_body/add/ReadVariableOploop_body/add/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
²
#__inference_signature_wrapper_38067
input_1
unknown:
	unknown_0:
	unknown_1:
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_37333o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¯

 __inference__wrapped_model_37333
input_1A
/model_tn__toy_loop_body_readvariableop_resource:G
1model_tn__toy_loop_body_readvariableop_1_resource:A
3model_tn__toy_loop_body_add_readvariableop_resource:
identity¢&model/tn__toy/loop_body/ReadVariableOp¢(model/tn__toy/loop_body/ReadVariableOp_1¢*model/tn__toy/loop_body/add/ReadVariableOpJ
model/tn__toy/ShapeShapeinput_1*
T0*
_output_shapes
:k
!model/tn__toy/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#model/tn__toy/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#model/tn__toy/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
model/tn__toy/strided_sliceStridedSlicemodel/tn__toy/Shape:output:0*model/tn__toy/strided_slice/stack:output:0,model/tn__toy/strided_slice/stack_1:output:0,model/tn__toy/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
model/tn__toy/Rank/packedPack$model/tn__toy/strided_slice:output:0*
N*
T0*
_output_shapes
:T
model/tn__toy/RankConst*
_output_shapes
: *
dtype0*
value	B :[
model/tn__toy/range/startConst*
_output_shapes
: *
dtype0*
value	B : [
model/tn__toy/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
model/tn__toy/rangeRange"model/tn__toy/range/start:output:0model/tn__toy/Rank:output:0"model/tn__toy/range/delta:output:0*
_output_shapes
:s
model/tn__toy/Max/inputPack$model/tn__toy/strided_slice:output:0*
N*
T0*
_output_shapes
:y
model/tn__toy/MaxMax model/tn__toy/Max/input:output:0model/tn__toy/range:output:0*
T0*
_output_shapes
: v
4model/tn__toy/loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : ¹
.model/tn__toy/loop_body/PlaceholderWithDefaultPlaceholderWithDefault=model/tn__toy/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: T
model/tn__toy/loop_body/ShapeShapeinput_1*
T0*
_output_shapes
:u
+model/tn__toy/loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-model/tn__toy/loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-model/tn__toy/loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%model/tn__toy/loop_body/strided_sliceStridedSlice&model/tn__toy/loop_body/Shape:output:04model/tn__toy/loop_body/strided_slice/stack:output:06model/tn__toy/loop_body/strided_slice/stack_1:output:06model/tn__toy/loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model/tn__toy/loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :§
model/tn__toy/loop_body/GreaterGreater.model/tn__toy/loop_body/strided_slice:output:0*model/tn__toy/loop_body/Greater/y:output:0*
T0*
_output_shapes
: d
"model/tn__toy/loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : Ø
 model/tn__toy/loop_body/SelectV2SelectV2#model/tn__toy/loop_body/Greater:z:07model/tn__toy/loop_body/PlaceholderWithDefault:output:0+model/tn__toy/loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: g
%model/tn__toy/loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ü
 model/tn__toy/loop_body/GatherV2GatherV2input_1)model/tn__toy/loop_body/SelectV2:output:0.model/tn__toy/loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes

:
&model/tn__toy/loop_body/ReadVariableOpReadVariableOp/model_tn__toy_loop_body_readvariableop_resource*
_output_shapes

:*
dtype0
(model/tn__toy/loop_body/ReadVariableOp_1ReadVariableOp1model_tn__toy_loop_body_readvariableop_1_resource*"
_output_shapes
:*
dtype0
0model/tn__toy/loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       Ç
+model/tn__toy/loop_body/Tensordot/transpose	Transpose)model/tn__toy/loop_body/GatherV2:output:09model/tn__toy/loop_body/Tensordot/transpose/perm:output:0*
T0*
_output_shapes

:¼
(model/tn__toy/loop_body/Tensordot/MatMulMatMul/model/tn__toy/loop_body/Tensordot/transpose:y:0.model/tn__toy/loop_body/ReadVariableOp:value:0*
T0*
_output_shapes

:
2model/tn__toy/loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ö
-model/tn__toy/loop_body/Tensordot_1/transpose	Transpose0model/tn__toy/loop_body/ReadVariableOp_1:value:0;model/tn__toy/loop_body/Tensordot_1/transpose/perm:output:0*
T0*"
_output_shapes
:
1model/tn__toy/loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Î
+model/tn__toy/loop_body/Tensordot_1/ReshapeReshape1model/tn__toy/loop_body/Tensordot_1/transpose:y:0:model/tn__toy/loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:
3model/tn__toy/loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ó
-model/tn__toy/loop_body/Tensordot_1/Reshape_1Reshape2model/tn__toy/loop_body/Tensordot/MatMul:product:0<model/tn__toy/loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:Ë
*model/tn__toy/loop_body/Tensordot_1/MatMulMatMul4model/tn__toy/loop_body/Tensordot_1/Reshape:output:06model/tn__toy/loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes

:s
)model/tn__toy/loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:½
#model/tn__toy/loop_body/Tensordot_1Reshape4model/tn__toy/loop_body/Tensordot_1/MatMul:product:02model/tn__toy/loop_body/Tensordot_1/shape:output:0*
T0*
_output_shapes
:
*model/tn__toy/loop_body/add/ReadVariableOpReadVariableOp3model_tn__toy_loop_body_add_readvariableop_resource*
_output_shapes
:*
dtype0«
model/tn__toy/loop_body/addAddV2,model/tn__toy/loop_body/Tensordot_1:output:02model/tn__toy/loop_body/add/ReadVariableOp:value:0*
T0*
_output_shapes
:j
 model/tn__toy/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
model/tn__toy/pfor/ReshapeReshapemodel/tn__toy/Max:output:0)model/tn__toy/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:`
model/tn__toy/pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : `
model/tn__toy/pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :´
model/tn__toy/pfor/rangeRange'model/tn__toy/pfor/range/start:output:0model/tn__toy/Max:output:0'model/tn__toy/pfor/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
*model/tn__toy/loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : m
+model/tn__toy/loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :¾
)model/tn__toy/loop_body/SelectV2/pfor/addAddV23model/tn__toy/loop_body/SelectV2/pfor/Rank:output:04model/tn__toy/loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: n
,model/tn__toy/loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :n
,model/tn__toy/loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : o
-model/tn__toy/loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ä
+model/tn__toy/loop_body/SelectV2/pfor/add_1AddV25model/tn__toy/loop_body/SelectV2/pfor/Rank_2:output:06model/tn__toy/loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: ¿
-model/tn__toy/loop_body/SelectV2/pfor/MaximumMaximum5model/tn__toy/loop_body/SelectV2/pfor/Rank_1:output:0-model/tn__toy/loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: ¿
/model/tn__toy/loop_body/SelectV2/pfor/Maximum_1Maximum/model/tn__toy/loop_body/SelectV2/pfor/add_1:z:01model/tn__toy/loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: |
+model/tn__toy/loop_body/SelectV2/pfor/ShapeShape!model/tn__toy/pfor/range:output:0*
T0*
_output_shapes
:½
)model/tn__toy/loop_body/SelectV2/pfor/subSub3model/tn__toy/loop_body/SelectV2/pfor/Maximum_1:z:05model/tn__toy/loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: }
3model/tn__toy/loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ê
-model/tn__toy/loop_body/SelectV2/pfor/ReshapeReshape-model/tn__toy/loop_body/SelectV2/pfor/sub:z:0<model/tn__toy/loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:z
0model/tn__toy/loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:È
*model/tn__toy/loop_body/SelectV2/pfor/TileTile9model/tn__toy/loop_body/SelectV2/pfor/Tile/input:output:06model/tn__toy/loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: 
9model/tn__toy/loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;model/tn__toy/loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;model/tn__toy/loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3model/tn__toy/loop_body/SelectV2/pfor/strided_sliceStridedSlice4model/tn__toy/loop_body/SelectV2/pfor/Shape:output:0Bmodel/tn__toy/loop_body/SelectV2/pfor/strided_slice/stack:output:0Dmodel/tn__toy/loop_body/SelectV2/pfor/strided_slice/stack_1:output:0Dmodel/tn__toy/loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
;model/tn__toy/loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
=model/tn__toy/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
=model/tn__toy/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5model/tn__toy/loop_body/SelectV2/pfor/strided_slice_1StridedSlice4model/tn__toy/loop_body/SelectV2/pfor/Shape:output:0Dmodel/tn__toy/loop_body/SelectV2/pfor/strided_slice_1/stack:output:0Fmodel/tn__toy/loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:0Fmodel/tn__toy/loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_masks
1model/tn__toy/loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Õ
,model/tn__toy/loop_body/SelectV2/pfor/concatConcatV2<model/tn__toy/loop_body/SelectV2/pfor/strided_slice:output:03model/tn__toy/loop_body/SelectV2/pfor/Tile:output:0>model/tn__toy/loop_body/SelectV2/pfor/strided_slice_1:output:0:model/tn__toy/loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Â
/model/tn__toy/loop_body/SelectV2/pfor/Reshape_1Reshape!model/tn__toy/pfor/range:output:05model/tn__toy/loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
.model/tn__toy/loop_body/SelectV2/pfor/SelectV2SelectV2#model/tn__toy/loop_body/Greater:z:08model/tn__toy/loop_body/SelectV2/pfor/Reshape_1:output:0+model/tn__toy/loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
3model/tn__toy/loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
.model/tn__toy/loop_body/GatherV2/pfor/GatherV2GatherV2input_17model/tn__toy/loop_body/SelectV2/pfor/SelectV2:output:0<model/tn__toy/loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
6model/tn__toy/loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Þ
4model/tn__toy/loop_body/Tensordot/transpose/pfor/addAddV29model/tn__toy/loop_body/Tensordot/transpose/perm:output:0?model/tn__toy/loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:
@model/tn__toy/loop_body/Tensordot/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: ~
<model/tn__toy/loop_body/Tensordot/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ½
7model/tn__toy/loop_body/Tensordot/transpose/pfor/concatConcatV2Imodel/tn__toy/loop_body/Tensordot/transpose/pfor/concat/values_0:output:08model/tn__toy/loop_body/Tensordot/transpose/pfor/add:z:0Emodel/tn__toy/loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ø
:model/tn__toy/loop_body/Tensordot/transpose/pfor/Transpose	Transpose7model/tn__toy/loop_body/GatherV2/pfor/GatherV2:output:0@model/tn__toy/loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
3model/tn__toy/loop_body/Tensordot/MatMul/pfor/ShapeShape>model/tn__toy/loop_body/Tensordot/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:
=model/tn__toy/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
3model/tn__toy/loop_body/Tensordot/MatMul/pfor/splitSplitFmodel/tn__toy/loop_body/Tensordot/MatMul/pfor/split/split_dim:output:0<model/tn__toy/loop_body/Tensordot/MatMul/pfor/Shape:output:0*
T0*&
_output_shapes
:::*
	num_split~
;model/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
=model/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB ç
5model/tn__toy/loop_body/Tensordot/MatMul/pfor/ReshapeReshape<model/tn__toy/loop_body/Tensordot/MatMul/pfor/split:output:0Fmodel/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1:output:0*
T0*
_output_shapes
: 
=model/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB 
?model/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ë
7model/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_1Reshape<model/tn__toy/loop_body/Tensordot/MatMul/pfor/split:output:1Hmodel/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: 
=model/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB 
?model/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ë
7model/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_2Reshape<model/tn__toy/loop_body/Tensordot/MatMul/pfor/split:output:2Hmodel/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: Û
1model/tn__toy/loop_body/Tensordot/MatMul/pfor/mulMul>model/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape:output:0@model/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: ì
=model/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack5model/tn__toy/loop_body/Tensordot/MatMul/pfor/mul:z:0@model/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:
7model/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape>model/tn__toy/loop_body/Tensordot/transpose/pfor/Transpose:y:0Fmodel/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿâ
4model/tn__toy/loop_body/Tensordot/MatMul/pfor/MatMulMatMul@model/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0.model/tn__toy/loop_body/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
?model/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¿
=model/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePack>model/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape:output:0@model/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0Hmodel/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:
7model/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_4Reshape>model/tn__toy/loop_body/Tensordot/MatMul/pfor/MatMul:product:0Fmodel/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>model/tn__toy/loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
9model/tn__toy/loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2#model/tn__toy/pfor/Reshape:output:0<model/tn__toy/loop_body/Tensordot_1/Reshape_1/shape:output:0Gmodel/tn__toy/loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
:model/tn__toy/loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape@model/tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_4:output:0Bmodel/tn__toy/loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
5model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/ShapeShapeCmodel/tn__toy/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
Cmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Emodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Emodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Á
=model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSlice>model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Lmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0Nmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0Nmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Emodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Gmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Gmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
?model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSlice>model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Nmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Pmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Pmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskõ
7model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/MinimumMinimumFmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0Hmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: y
7model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :Þ
5model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/EqualEqual;model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Minimum:z:0@model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: 
8model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          
8model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          ¦
6model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/SelectSelect9model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Equal:z:0Amodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Select/t:output:0Amodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:
Emodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
Gmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Gmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
?model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSlice>model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Nmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Pmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Pmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Emodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Gmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Gmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
?model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSlice>model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Nmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Pmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Pmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Emodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
Gmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Gmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
?model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSlice>model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Nmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Pmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Pmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÉ
5model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/stackPackHmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0Hmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0Hmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:
9model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/transpose	TransposeCmodel/tn__toy/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0?model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ÷
7model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/ReshapeReshape=model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/transpose:y:0>model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
7model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Shape_1Shape@model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:
?model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
5model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/splitSplitHmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/split/split_dim:output:0@model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split
?model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Amodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ñ
9model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_1Reshape>model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/split:output:0Jmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: 
?model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Amodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ñ
9model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_2Reshape>model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/split:output:1Jmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: 
?model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Amodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB ñ
9model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_3Reshape>model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/split:output:2Jmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ã
3model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/mulMulBmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0Bmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ò
?model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePackBmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:07model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:
9model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_4Reshape@model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0Hmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
6model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul4model/tn__toy/loop_body/Tensordot_1/Reshape:output:0Bmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Amodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
?model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePackJmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:0Bmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0Bmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:
9model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_5Reshape@model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0Hmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
@model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
;model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/transpose_1	TransposeBmodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0Imodel/tn__toy/loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
4model/tn__toy/loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
/model/tn__toy/loop_body/Tensordot_1/pfor/concatConcatV2#model/tn__toy/pfor/Reshape:output:02model/tn__toy/loop_body/Tensordot_1/shape:output:0=model/tn__toy/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:è
0model/tn__toy/loop_body/Tensordot_1/pfor/ReshapeReshape?model/tn__toy/loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:08model/tn__toy/loop_body/Tensordot_1/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
%model/tn__toy/loop_body/add/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :i
'model/tn__toy/loop_body/add/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :h
&model/tn__toy/loop_body/add/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :±
$model/tn__toy/loop_body/add/pfor/addAddV20model/tn__toy/loop_body/add/pfor/Rank_1:output:0/model/tn__toy/loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: ®
(model/tn__toy/loop_body/add/pfor/MaximumMaximum(model/tn__toy/loop_body/add/pfor/add:z:0.model/tn__toy/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: 
&model/tn__toy/loop_body/add/pfor/ShapeShape9model/tn__toy/loop_body/Tensordot_1/pfor/Reshape:output:0*
T0*
_output_shapes
:ª
$model/tn__toy/loop_body/add/pfor/subSub,model/tn__toy/loop_body/add/pfor/Maximum:z:0.model/tn__toy/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: x
.model/tn__toy/loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:»
(model/tn__toy/loop_body/add/pfor/ReshapeReshape(model/tn__toy/loop_body/add/pfor/sub:z:07model/tn__toy/loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:u
+model/tn__toy/loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:¹
%model/tn__toy/loop_body/add/pfor/TileTile4model/tn__toy/loop_body/add/pfor/Tile/input:output:01model/tn__toy/loop_body/add/pfor/Reshape:output:0*
T0*
_output_shapes
: ~
4model/tn__toy/loop_body/add/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6model/tn__toy/loop_body/add/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6model/tn__toy/loop_body/add/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
.model/tn__toy/loop_body/add/pfor/strided_sliceStridedSlice/model/tn__toy/loop_body/add/pfor/Shape:output:0=model/tn__toy/loop_body/add/pfor/strided_slice/stack:output:0?model/tn__toy/loop_body/add/pfor/strided_slice/stack_1:output:0?model/tn__toy/loop_body/add/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
6model/tn__toy/loop_body/add/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
8model/tn__toy/loop_body/add/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
8model/tn__toy/loop_body/add/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ú
0model/tn__toy/loop_body/add/pfor/strided_slice_1StridedSlice/model/tn__toy/loop_body/add/pfor/Shape:output:0?model/tn__toy/loop_body/add/pfor/strided_slice_1/stack:output:0Amodel/tn__toy/loop_body/add/pfor/strided_slice_1/stack_1:output:0Amodel/tn__toy/loop_body/add/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskn
,model/tn__toy/loop_body/add/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
'model/tn__toy/loop_body/add/pfor/concatConcatV27model/tn__toy/loop_body/add/pfor/strided_slice:output:0.model/tn__toy/loop_body/add/pfor/Tile:output:09model/tn__toy/loop_body/add/pfor/strided_slice_1:output:05model/tn__toy/loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ô
*model/tn__toy/loop_body/add/pfor/Reshape_1Reshape9model/tn__toy/loop_body/Tensordot_1/pfor/Reshape:output:00model/tn__toy/loop_body/add/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
&model/tn__toy/loop_body/add/pfor/AddV2AddV23model/tn__toy/loop_body/add/pfor/Reshape_1:output:02model/tn__toy/loop_body/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
model/tn__toy/SoftmaxSoftmax*model/tn__toy/loop_body/add/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
IdentityIdentitymodel/tn__toy/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
NoOpNoOp'^model/tn__toy/loop_body/ReadVariableOp)^model/tn__toy/loop_body/ReadVariableOp_1+^model/tn__toy/loop_body/add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2P
&model/tn__toy/loop_body/ReadVariableOp&model/tn__toy/loop_body/ReadVariableOp2T
(model/tn__toy/loop_body/ReadVariableOp_1(model/tn__toy/loop_body/ReadVariableOp_12X
*model/tn__toy/loop_body/add/ReadVariableOp*model/tn__toy/loop_body/add/ReadVariableOp:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
 
·
__inference__traced_save_38332
file_prefix#
savev2_mps1_read_readvariableop#
savev2_mps2_read_readvariableop#
savev2_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ñ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ú
valueðBíB4layer_with_weights-0/mps1/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/mps2/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B Û
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mps1_read_readvariableopsavev2_mps2_read_readvariableopsavev2_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*M
_input_shapes<
:: :::: : : : : : ::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

::($
"
_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: : 


_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
Ó-

!__inference__traced_restore_38375
file_prefix'
assignvariableop_mps1:-
assignvariableop_1_mps2:%
assignvariableop_2_bias:%
assignvariableop_3_sgd_iter:	 &
assignvariableop_4_sgd_decay: .
$assignvariableop_5_sgd_learning_rate: )
assignvariableop_6_sgd_momentum: "
assignvariableop_7_total: "
assignvariableop_8_count: /
!assignvariableop_9_true_positives:1
#assignvariableop_10_false_positives:
identity_12¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ô
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ú
valueðBíB4layer_with_weights-0/mps1/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/mps2/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B Ú
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*D
_output_shapes2
0::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_mps1Identity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_mps2Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_sgd_iterIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_sgd_decayIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp$assignvariableop_5_sgd_learning_rateIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_sgd_momentumIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_totalIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_countIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp!assignvariableop_9_true_positivesIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp#assignvariableop_10_false_positivesIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Á
Identity_11Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_12IdentityIdentity_11:output:0^NoOp_1*
T0*
_output_shapes
: ®
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_12Identity_12:output:0*+
_input_shapes
: : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
·û

@__inference_model_layer_call_and_return_conditional_losses_38054

inputs;
)tn__toy_loop_body_readvariableop_resource:A
+tn__toy_loop_body_readvariableop_1_resource:;
-tn__toy_loop_body_add_readvariableop_resource:
identity¢ tn__toy/loop_body/ReadVariableOp¢"tn__toy/loop_body/ReadVariableOp_1¢$tn__toy/loop_body/add/ReadVariableOpC
tn__toy/ShapeShapeinputs*
T0*
_output_shapes
:e
tn__toy/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
tn__toy/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
tn__toy/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
tn__toy/strided_sliceStridedSlicetn__toy/Shape:output:0$tn__toy/strided_slice/stack:output:0&tn__toy/strided_slice/stack_1:output:0&tn__toy/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
tn__toy/Rank/packedPacktn__toy/strided_slice:output:0*
N*
T0*
_output_shapes
:N
tn__toy/RankConst*
_output_shapes
: *
dtype0*
value	B :U
tn__toy/range/startConst*
_output_shapes
: *
dtype0*
value	B : U
tn__toy/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
tn__toy/rangeRangetn__toy/range/start:output:0tn__toy/Rank:output:0tn__toy/range/delta:output:0*
_output_shapes
:g
tn__toy/Max/inputPacktn__toy/strided_slice:output:0*
N*
T0*
_output_shapes
:g
tn__toy/MaxMaxtn__toy/Max/input:output:0tn__toy/range:output:0*
T0*
_output_shapes
: p
.tn__toy/loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : ­
(tn__toy/loop_body/PlaceholderWithDefaultPlaceholderWithDefault7tn__toy/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: M
tn__toy/loop_body/ShapeShapeinputs*
T0*
_output_shapes
:o
%tn__toy/loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'tn__toy/loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'tn__toy/loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
tn__toy/loop_body/strided_sliceStridedSlice tn__toy/loop_body/Shape:output:0.tn__toy/loop_body/strided_slice/stack:output:00tn__toy/loop_body/strided_slice/stack_1:output:00tn__toy/loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
tn__toy/loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :
tn__toy/loop_body/GreaterGreater(tn__toy/loop_body/strided_slice:output:0$tn__toy/loop_body/Greater/y:output:0*
T0*
_output_shapes
: ^
tn__toy/loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : À
tn__toy/loop_body/SelectV2SelectV2tn__toy/loop_body/Greater:z:01tn__toy/loop_body/PlaceholderWithDefault:output:0%tn__toy/loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: a
tn__toy/loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : É
tn__toy/loop_body/GatherV2GatherV2inputs#tn__toy/loop_body/SelectV2:output:0(tn__toy/loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes

:
 tn__toy/loop_body/ReadVariableOpReadVariableOp)tn__toy_loop_body_readvariableop_resource*
_output_shapes

:*
dtype0
"tn__toy/loop_body/ReadVariableOp_1ReadVariableOp+tn__toy_loop_body_readvariableop_1_resource*"
_output_shapes
:*
dtype0{
*tn__toy/loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       µ
%tn__toy/loop_body/Tensordot/transpose	Transpose#tn__toy/loop_body/GatherV2:output:03tn__toy/loop_body/Tensordot/transpose/perm:output:0*
T0*
_output_shapes

:ª
"tn__toy/loop_body/Tensordot/MatMulMatMul)tn__toy/loop_body/Tensordot/transpose:y:0(tn__toy/loop_body/ReadVariableOp:value:0*
T0*
_output_shapes

:
,tn__toy/loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ä
'tn__toy/loop_body/Tensordot_1/transpose	Transpose*tn__toy/loop_body/ReadVariableOp_1:value:05tn__toy/loop_body/Tensordot_1/transpose/perm:output:0*
T0*"
_output_shapes
:|
+tn__toy/loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ¼
%tn__toy/loop_body/Tensordot_1/ReshapeReshape+tn__toy/loop_body/Tensordot_1/transpose:y:04tn__toy/loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:~
-tn__toy/loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Á
'tn__toy/loop_body/Tensordot_1/Reshape_1Reshape,tn__toy/loop_body/Tensordot/MatMul:product:06tn__toy/loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:¹
$tn__toy/loop_body/Tensordot_1/MatMulMatMul.tn__toy/loop_body/Tensordot_1/Reshape:output:00tn__toy/loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes

:m
#tn__toy/loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:«
tn__toy/loop_body/Tensordot_1Reshape.tn__toy/loop_body/Tensordot_1/MatMul:product:0,tn__toy/loop_body/Tensordot_1/shape:output:0*
T0*
_output_shapes
:
$tn__toy/loop_body/add/ReadVariableOpReadVariableOp-tn__toy_loop_body_add_readvariableop_resource*
_output_shapes
:*
dtype0
tn__toy/loop_body/addAddV2&tn__toy/loop_body/Tensordot_1:output:0,tn__toy/loop_body/add/ReadVariableOp:value:0*
T0*
_output_shapes
:d
tn__toy/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
tn__toy/pfor/ReshapeReshapetn__toy/Max:output:0#tn__toy/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:Z
tn__toy/pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : Z
tn__toy/pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
tn__toy/pfor/rangeRange!tn__toy/pfor/range/start:output:0tn__toy/Max:output:0!tn__toy/pfor/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$tn__toy/loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : g
%tn__toy/loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :¬
#tn__toy/loop_body/SelectV2/pfor/addAddV2-tn__toy/loop_body/SelectV2/pfor/Rank:output:0.tn__toy/loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: h
&tn__toy/loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :h
&tn__toy/loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : i
'tn__toy/loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :²
%tn__toy/loop_body/SelectV2/pfor/add_1AddV2/tn__toy/loop_body/SelectV2/pfor/Rank_2:output:00tn__toy/loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: ­
'tn__toy/loop_body/SelectV2/pfor/MaximumMaximum/tn__toy/loop_body/SelectV2/pfor/Rank_1:output:0'tn__toy/loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: ­
)tn__toy/loop_body/SelectV2/pfor/Maximum_1Maximum)tn__toy/loop_body/SelectV2/pfor/add_1:z:0+tn__toy/loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: p
%tn__toy/loop_body/SelectV2/pfor/ShapeShapetn__toy/pfor/range:output:0*
T0*
_output_shapes
:«
#tn__toy/loop_body/SelectV2/pfor/subSub-tn__toy/loop_body/SelectV2/pfor/Maximum_1:z:0/tn__toy/loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: w
-tn__toy/loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:¸
'tn__toy/loop_body/SelectV2/pfor/ReshapeReshape'tn__toy/loop_body/SelectV2/pfor/sub:z:06tn__toy/loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:t
*tn__toy/loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:¶
$tn__toy/loop_body/SelectV2/pfor/TileTile3tn__toy/loop_body/SelectV2/pfor/Tile/input:output:00tn__toy/loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: }
3tn__toy/loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tn__toy/loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tn__toy/loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
-tn__toy/loop_body/SelectV2/pfor/strided_sliceStridedSlice.tn__toy/loop_body/SelectV2/pfor/Shape:output:0<tn__toy/loop_body/SelectV2/pfor/strided_slice/stack:output:0>tn__toy/loop_body/SelectV2/pfor/strided_slice/stack_1:output:0>tn__toy/loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
5tn__toy/loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
7tn__toy/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
7tn__toy/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ó
/tn__toy/loop_body/SelectV2/pfor/strided_slice_1StridedSlice.tn__toy/loop_body/SelectV2/pfor/Shape:output:0>tn__toy/loop_body/SelectV2/pfor/strided_slice_1/stack:output:0@tn__toy/loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:0@tn__toy/loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskm
+tn__toy/loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
&tn__toy/loop_body/SelectV2/pfor/concatConcatV26tn__toy/loop_body/SelectV2/pfor/strided_slice:output:0-tn__toy/loop_body/SelectV2/pfor/Tile:output:08tn__toy/loop_body/SelectV2/pfor/strided_slice_1:output:04tn__toy/loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:°
)tn__toy/loop_body/SelectV2/pfor/Reshape_1Reshapetn__toy/pfor/range:output:0/tn__toy/loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
(tn__toy/loop_body/SelectV2/pfor/SelectV2SelectV2tn__toy/loop_body/Greater:z:02tn__toy/loop_body/SelectV2/pfor/Reshape_1:output:0%tn__toy/loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
-tn__toy/loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
(tn__toy/loop_body/GatherV2/pfor/GatherV2GatherV2inputs1tn__toy/loop_body/SelectV2/pfor/SelectV2:output:06tn__toy/loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0tn__toy/loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ì
.tn__toy/loop_body/Tensordot/transpose/pfor/addAddV23tn__toy/loop_body/Tensordot/transpose/perm:output:09tn__toy/loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:
:tn__toy/loop_body/Tensordot/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: x
6tn__toy/loop_body/Tensordot/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¥
1tn__toy/loop_body/Tensordot/transpose/pfor/concatConcatV2Ctn__toy/loop_body/Tensordot/transpose/pfor/concat/values_0:output:02tn__toy/loop_body/Tensordot/transpose/pfor/add:z:0?tn__toy/loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:æ
4tn__toy/loop_body/Tensordot/transpose/pfor/Transpose	Transpose1tn__toy/loop_body/GatherV2/pfor/GatherV2:output:0:tn__toy/loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-tn__toy/loop_body/Tensordot/MatMul/pfor/ShapeShape8tn__toy/loop_body/Tensordot/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:y
7tn__toy/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ò
-tn__toy/loop_body/Tensordot/MatMul/pfor/splitSplit@tn__toy/loop_body/Tensordot/MatMul/pfor/split/split_dim:output:06tn__toy/loop_body/Tensordot/MatMul/pfor/Shape:output:0*
T0*&
_output_shapes
:::*
	num_splitx
5tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB z
7tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB Õ
/tn__toy/loop_body/Tensordot/MatMul/pfor/ReshapeReshape6tn__toy/loop_body/Tensordot/MatMul/pfor/split:output:0@tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1:output:0*
T0*
_output_shapes
: z
7tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB |
9tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB Ù
1tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_1Reshape6tn__toy/loop_body/Tensordot/MatMul/pfor/split:output:1Btn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: z
7tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB |
9tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB Ù
1tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_2Reshape6tn__toy/loop_body/Tensordot/MatMul/pfor/split:output:2Btn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: É
+tn__toy/loop_body/Tensordot/MatMul/pfor/mulMul8tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape:output:0:tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: Ú
7tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack/tn__toy/loop_body/Tensordot/MatMul/pfor/mul:z:0:tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:ó
1tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape8tn__toy/loop_body/Tensordot/transpose/pfor/Transpose:y:0@tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÐ
.tn__toy/loop_body/Tensordot/MatMul/pfor/MatMulMatMul:tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0(tn__toy/loop_body/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
9tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ§
7tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePack8tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape:output:0:tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0Btn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:î
1tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_4Reshape8tn__toy/loop_body/Tensordot/MatMul/pfor/MatMul:product:0@tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
8tn__toy/loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
3tn__toy/loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2tn__toy/pfor/Reshape:output:06tn__toy/loop_body/Tensordot_1/Reshape_1/shape:output:0Atn__toy/loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ï
4tn__toy/loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape:tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_4:output:0<tn__toy/loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/tn__toy/loop_body/Tensordot_1/MatMul/pfor/ShapeShape=tn__toy/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
=tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:£
7tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSlice8tn__toy/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Ftn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0Htn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0Htn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Atn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Atn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
9tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSlice8tn__toy/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Htn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Jtn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Jtn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskã
1tn__toy/loop_body/Tensordot_1/MatMul/pfor/MinimumMinimum@tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0Btn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: s
1tn__toy/loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :Ì
/tn__toy/loop_body/Tensordot_1/MatMul/pfor/EqualEqual5tn__toy/loop_body/Tensordot_1/MatMul/pfor/Minimum:z:0:tn__toy/loop_body/Tensordot_1/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: 
2tn__toy/loop_body/Tensordot_1/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          
2tn__toy/loop_body/Tensordot_1/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          
0tn__toy/loop_body/Tensordot_1/MatMul/pfor/SelectSelect3tn__toy/loop_body/Tensordot_1/MatMul/pfor/Equal:z:0;tn__toy/loop_body/Tensordot_1/MatMul/pfor/Select/t:output:0;tn__toy/loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:
?tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
Atn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Atn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
9tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSlice8tn__toy/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Htn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Jtn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Jtn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Atn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Atn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
9tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSlice8tn__toy/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Htn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Jtn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Jtn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
Atn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Atn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
9tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSlice8tn__toy/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Htn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Jtn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Jtn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask±
/tn__toy/loop_body/Tensordot_1/MatMul/pfor/stackPackBtn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0Btn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0Btn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:
3tn__toy/loop_body/Tensordot_1/MatMul/pfor/transpose	Transpose=tn__toy/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:09tn__toy/loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿå
1tn__toy/loop_body/Tensordot_1/MatMul/pfor/ReshapeReshape7tn__toy/loop_body/Tensordot_1/MatMul/pfor/transpose:y:08tn__toy/loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1tn__toy/loop_body/Tensordot_1/MatMul/pfor/Shape_1Shape:tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:{
9tn__toy/loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ú
/tn__toy/loop_body/Tensordot_1/MatMul/pfor/splitSplitBtn__toy/loop_body/Tensordot_1/MatMul/pfor/split/split_dim:output:0:tn__toy/loop_body/Tensordot_1/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split|
9tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ~
;tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ß
3tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_1Reshape8tn__toy/loop_body/Tensordot_1/MatMul/pfor/split:output:0Dtn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: |
9tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ~
;tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ß
3tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_2Reshape8tn__toy/loop_body/Tensordot_1/MatMul/pfor/split:output:1Dtn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: |
9tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB ~
;tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB ß
3tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_3Reshape8tn__toy/loop_body/Tensordot_1/MatMul/pfor/split:output:2Dtn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: Ñ
-tn__toy/loop_body/Tensordot_1/MatMul/pfor/mulMul<tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0<tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: à
9tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePack<tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:01tn__toy/loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:ù
3tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_4Reshape:tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0Btn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
0tn__toy/loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul.tn__toy/loop_body/Tensordot_1/Reshape:output:0<tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
;tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ±
9tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePackDtn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:0<tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0<tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:
3tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_5Reshape:tn__toy/loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0Btn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:tn__toy/loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          û
5tn__toy/loop_body/Tensordot_1/MatMul/pfor/transpose_1	Transpose<tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0Ctn__toy/loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
.tn__toy/loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : é
)tn__toy/loop_body/Tensordot_1/pfor/concatConcatV2tn__toy/pfor/Reshape:output:0,tn__toy/loop_body/Tensordot_1/shape:output:07tn__toy/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ö
*tn__toy/loop_body/Tensordot_1/pfor/ReshapeReshape9tn__toy/loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:02tn__toy/loop_body/Tensordot_1/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
tn__toy/loop_body/add/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :c
!tn__toy/loop_body/add/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :b
 tn__toy/loop_body/add/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :
tn__toy/loop_body/add/pfor/addAddV2*tn__toy/loop_body/add/pfor/Rank_1:output:0)tn__toy/loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: 
"tn__toy/loop_body/add/pfor/MaximumMaximum"tn__toy/loop_body/add/pfor/add:z:0(tn__toy/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: 
 tn__toy/loop_body/add/pfor/ShapeShape3tn__toy/loop_body/Tensordot_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
tn__toy/loop_body/add/pfor/subSub&tn__toy/loop_body/add/pfor/Maximum:z:0(tn__toy/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: r
(tn__toy/loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:©
"tn__toy/loop_body/add/pfor/ReshapeReshape"tn__toy/loop_body/add/pfor/sub:z:01tn__toy/loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:o
%tn__toy/loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:§
tn__toy/loop_body/add/pfor/TileTile.tn__toy/loop_body/add/pfor/Tile/input:output:0+tn__toy/loop_body/add/pfor/Reshape:output:0*
T0*
_output_shapes
: x
.tn__toy/loop_body/add/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0tn__toy/loop_body/add/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0tn__toy/loop_body/add/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ö
(tn__toy/loop_body/add/pfor/strided_sliceStridedSlice)tn__toy/loop_body/add/pfor/Shape:output:07tn__toy/loop_body/add/pfor/strided_slice/stack:output:09tn__toy/loop_body/add/pfor/strided_slice/stack_1:output:09tn__toy/loop_body/add/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskz
0tn__toy/loop_body/add/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:|
2tn__toy/loop_body/add/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2tn__toy/loop_body/add/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ü
*tn__toy/loop_body/add/pfor/strided_slice_1StridedSlice)tn__toy/loop_body/add/pfor/Shape:output:09tn__toy/loop_body/add/pfor/strided_slice_1/stack:output:0;tn__toy/loop_body/add/pfor/strided_slice_1/stack_1:output:0;tn__toy/loop_body/add/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskh
&tn__toy/loop_body/add/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
!tn__toy/loop_body/add/pfor/concatConcatV21tn__toy/loop_body/add/pfor/strided_slice:output:0(tn__toy/loop_body/add/pfor/Tile:output:03tn__toy/loop_body/add/pfor/strided_slice_1:output:0/tn__toy/loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Â
$tn__toy/loop_body/add/pfor/Reshape_1Reshape3tn__toy/loop_body/Tensordot_1/pfor/Reshape:output:0*tn__toy/loop_body/add/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
 tn__toy/loop_body/add/pfor/AddV2AddV2-tn__toy/loop_body/add/pfor/Reshape_1:output:0,tn__toy/loop_body/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
tn__toy/SoftmaxSoftmax$tn__toy/loop_body/add/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitytn__toy/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
NoOpNoOp!^tn__toy/loop_body/ReadVariableOp#^tn__toy/loop_body/ReadVariableOp_1%^tn__toy/loop_body/add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2D
 tn__toy/loop_body/ReadVariableOp tn__toy/loop_body/ReadVariableOp2H
"tn__toy/loop_body/ReadVariableOp_1"tn__toy/loop_body/ReadVariableOp_12L
$tn__toy/loop_body/add/ReadVariableOp$tn__toy/loop_body/add/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
å
@__inference_model_layer_call_and_return_conditional_losses_37617
input_1
tn__toy_37609:#
tn__toy_37611:
tn__toy_37613:
identity¢tn__toy/StatefulPartitionedCallþ
tn__toy/StatefulPartitionedCallStatefulPartitionedCallinput_1tn__toy_37609tn__toy_37611tn__toy_37613*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_tn__toy_layer_call_and_return_conditional_losses_37532w
IdentityIdentity(tn__toy/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
NoOpNoOp ^tn__toy/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2B
tn__toy/StatefulPartitionedCalltn__toy/StatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Þ
ä
@__inference_model_layer_call_and_return_conditional_losses_37586

inputs
tn__toy_37578:#
tn__toy_37580:
tn__toy_37582:
identity¢tn__toy/StatefulPartitionedCallý
tn__toy/StatefulPartitionedCallStatefulPartitionedCallinputstn__toy_37578tn__toy_37580tn__toy_37582*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_tn__toy_layer_call_and_return_conditional_losses_37532w
IdentityIdentity(tn__toy/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
NoOpNoOp ^tn__toy/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2B
tn__toy/StatefulPartitionedCalltn__toy/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
µ
'__inference_tn__toy_layer_call_fn_38084

inputs
unknown:
	unknown_0:
	unknown_1:
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_tn__toy_layer_call_and_return_conditional_losses_37532o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
å
@__inference_model_layer_call_and_return_conditional_losses_37628
input_1
tn__toy_37620:#
tn__toy_37622:
tn__toy_37624:
identity¢tn__toy/StatefulPartitionedCallþ
tn__toy/StatefulPartitionedCallStatefulPartitionedCallinput_1tn__toy_37620tn__toy_37622tn__toy_37624*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_tn__toy_layer_call_and_return_conditional_losses_37532w
IdentityIdentity(tn__toy/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
NoOpNoOp ^tn__toy/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2B
tn__toy/StatefulPartitionedCalltn__toy/StatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
õ
³
%__inference_model_layer_call_fn_37666

inputs
unknown:
	unknown_0:
	unknown_1:
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_37586o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
Ý
B__inference_tn__toy_layer_call_and_return_conditional_losses_38276

inputs3
!loop_body_readvariableop_resource:9
#loop_body_readvariableop_1_resource:3
%loop_body_add_readvariableop_resource:
identity¢loop_body/ReadVariableOp¢loop_body/ReadVariableOp_1¢loop_body/add/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
Rank/packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:W
	Max/inputPackstrided_slice:output:0*
N*
T0*
_output_shapes
:O
MaxMaxMax/input:output:0range:output:0*
T0*
_output_shapes
: h
&loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : 
 loop_body/PlaceholderWithDefaultPlaceholderWithDefault/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: E
loop_body/ShapeShapeinputs*
T0*
_output_shapes
:g
loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
loop_body/strided_sliceStridedSliceloop_body/Shape:output:0&loop_body/strided_slice/stack:output:0(loop_body/strided_slice/stack_1:output:0(loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :}
loop_body/GreaterGreater loop_body/strided_slice:output:0loop_body/Greater/y:output:0*
T0*
_output_shapes
: V
loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B :  
loop_body/SelectV2SelectV2loop_body/Greater:z:0)loop_body/PlaceholderWithDefault:output:0loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: Y
loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ±
loop_body/GatherV2GatherV2inputsloop_body/SelectV2:output:0 loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes

:z
loop_body/ReadVariableOpReadVariableOp!loop_body_readvariableop_resource*
_output_shapes

:*
dtype0
loop_body/ReadVariableOp_1ReadVariableOp#loop_body_readvariableop_1_resource*"
_output_shapes
:*
dtype0s
"loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       
loop_body/Tensordot/transpose	Transposeloop_body/GatherV2:output:0+loop_body/Tensordot/transpose/perm:output:0*
T0*
_output_shapes

:
loop_body/Tensordot/MatMulMatMul!loop_body/Tensordot/transpose:y:0 loop_body/ReadVariableOp:value:0*
T0*
_output_shapes

:y
$loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¬
loop_body/Tensordot_1/transpose	Transpose"loop_body/ReadVariableOp_1:value:0-loop_body/Tensordot_1/transpose/perm:output:0*
T0*"
_output_shapes
:t
#loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ¤
loop_body/Tensordot_1/ReshapeReshape#loop_body/Tensordot_1/transpose:y:0,loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:v
%loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ©
loop_body/Tensordot_1/Reshape_1Reshape$loop_body/Tensordot/MatMul:product:0.loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:¡
loop_body/Tensordot_1/MatMulMatMul&loop_body/Tensordot_1/Reshape:output:0(loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes

:e
loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
loop_body/Tensordot_1Reshape&loop_body/Tensordot_1/MatMul:product:0$loop_body/Tensordot_1/shape:output:0*
T0*
_output_shapes
:~
loop_body/add/ReadVariableOpReadVariableOp%loop_body_add_readvariableop_resource*
_output_shapes
:*
dtype0
loop_body/addAddV2loop_body/Tensordot_1:output:0$loop_body/add/ReadVariableOp:value:0*
T0*
_output_shapes
:\
pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:g
pfor/ReshapeReshapeMax:output:0pfor/Reshape/shape:output:0*
T0*
_output_shapes
:R
pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : R
pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :|

pfor/rangeRangepfor/range/start:output:0Max:output:0pfor/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : _
loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :
loop_body/SelectV2/pfor/addAddV2%loop_body/SelectV2/pfor/Rank:output:0&loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: `
loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :`
loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : a
loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
loop_body/SelectV2/pfor/add_1AddV2'loop_body/SelectV2/pfor/Rank_2:output:0(loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: 
loop_body/SelectV2/pfor/MaximumMaximum'loop_body/SelectV2/pfor/Rank_1:output:0loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: 
!loop_body/SelectV2/pfor/Maximum_1Maximum!loop_body/SelectV2/pfor/add_1:z:0#loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: `
loop_body/SelectV2/pfor/ShapeShapepfor/range:output:0*
T0*
_output_shapes
:
loop_body/SelectV2/pfor/subSub%loop_body/SelectV2/pfor/Maximum_1:z:0'loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: o
%loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: 
loop_body/SelectV2/pfor/ReshapeReshapeloop_body/SelectV2/pfor/sub:z:0.loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:l
"loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:
loop_body/SelectV2/pfor/TileTile+loop_body/SelectV2/pfor/Tile/input:output:0(loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: u
+loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
%loop_body/SelectV2/pfor/strided_sliceStridedSlice&loop_body/SelectV2/pfor/Shape:output:04loop_body/SelectV2/pfor/strided_slice/stack:output:06loop_body/SelectV2/pfor/strided_slice/stack_1:output:06loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskw
-loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:y
/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: y
/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ë
'loop_body/SelectV2/pfor/strided_slice_1StridedSlice&loop_body/SelectV2/pfor/Shape:output:06loop_body/SelectV2/pfor/strided_slice_1/stack:output:08loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:08loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maske
#loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
loop_body/SelectV2/pfor/concatConcatV2.loop_body/SelectV2/pfor/strided_slice:output:0%loop_body/SelectV2/pfor/Tile:output:00loop_body/SelectV2/pfor/strided_slice_1:output:0,loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
!loop_body/SelectV2/pfor/Reshape_1Reshapepfor/range:output:0'loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
 loop_body/SelectV2/pfor/SelectV2SelectV2loop_body/Greater:z:0*loop_body/SelectV2/pfor/Reshape_1:output:0loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
%loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : è
 loop_body/GatherV2/pfor/GatherV2GatherV2inputs)loop_body/SelectV2/pfor/SelectV2:output:0.loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
(loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :´
&loop_body/Tensordot/transpose/pfor/addAddV2+loop_body/Tensordot/transpose/perm:output:01loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:|
2loop_body/Tensordot/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: p
.loop_body/Tensordot/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
)loop_body/Tensordot/transpose/pfor/concatConcatV2;loop_body/Tensordot/transpose/pfor/concat/values_0:output:0*loop_body/Tensordot/transpose/pfor/add:z:07loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Î
,loop_body/Tensordot/transpose/pfor/Transpose	Transpose)loop_body/GatherV2/pfor/GatherV2:output:02loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%loop_body/Tensordot/MatMul/pfor/ShapeShape0loop_body/Tensordot/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:q
/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ú
%loop_body/Tensordot/MatMul/pfor/splitSplit8loop_body/Tensordot/MatMul/pfor/split/split_dim:output:0.loop_body/Tensordot/MatMul/pfor/Shape:output:0*
T0*&
_output_shapes
:::*
	num_splitp
-loop_body/Tensordot/MatMul/pfor/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB r
/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB ½
'loop_body/Tensordot/MatMul/pfor/ReshapeReshape.loop_body/Tensordot/MatMul/pfor/split:output:08loop_body/Tensordot/MatMul/pfor/Reshape/shape_1:output:0*
T0*
_output_shapes
: r
/loop_body/Tensordot/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB t
1loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB Á
)loop_body/Tensordot/MatMul/pfor/Reshape_1Reshape.loop_body/Tensordot/MatMul/pfor/split:output:1:loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: r
/loop_body/Tensordot/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB t
1loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB Á
)loop_body/Tensordot/MatMul/pfor/Reshape_2Reshape.loop_body/Tensordot/MatMul/pfor/split:output:2:loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ±
#loop_body/Tensordot/MatMul/pfor/mulMul0loop_body/Tensordot/MatMul/pfor/Reshape:output:02loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: Â
/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack'loop_body/Tensordot/MatMul/pfor/mul:z:02loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:Û
)loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape0loop_body/Tensordot/transpose/pfor/Transpose:y:08loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸
&loop_body/Tensordot/MatMul/pfor/MatMulMatMul2loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0 loop_body/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
1loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePack0loop_body/Tensordot/MatMul/pfor/Reshape:output:02loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0:loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:Ö
)loop_body/Tensordot/MatMul/pfor/Reshape_4Reshape0loop_body/Tensordot/MatMul/pfor/MatMul:product:08loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
+loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2pfor/Reshape:output:0.loop_body/Tensordot_1/Reshape_1/shape:output:09loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:×
,loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape2loop_body/Tensordot/MatMul/pfor/Reshape_4:output:04loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'loop_body/Tensordot_1/MatMul/pfor/ShapeShape5loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
5loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:û
/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0>loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskË
)loop_body/Tensordot_1/MatMul/pfor/MinimumMinimum8loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: k
)loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :´
'loop_body/Tensordot_1/MatMul/pfor/EqualEqual-loop_body/Tensordot_1/MatMul/pfor/Minimum:z:02loop_body/Tensordot_1/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: 
*loop_body/Tensordot_1/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          
*loop_body/Tensordot_1/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          î
(loop_body/Tensordot_1/MatMul/pfor/SelectSelect+loop_body/Tensordot_1/MatMul/pfor/Equal:z:03loop_body/Tensordot_1/MatMul/pfor/Select/t:output:03loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
'loop_body/Tensordot_1/MatMul/pfor/stackPack:loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:ê
+loop_body/Tensordot_1/MatMul/pfor/transpose	Transpose5loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:01loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÍ
)loop_body/Tensordot_1/MatMul/pfor/ReshapeReshape/loop_body/Tensordot_1/MatMul/pfor/transpose:y:00loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)loop_body/Tensordot_1/MatMul/pfor/Shape_1Shape2loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:s
1loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : â
'loop_body/Tensordot_1/MatMul/pfor/splitSplit:loop_body/Tensordot_1/MatMul/pfor/split/split_dim:output:02loop_body/Tensordot_1/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_splitt
1loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB v
3loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB Ç
+loop_body/Tensordot_1/MatMul/pfor/Reshape_1Reshape0loop_body/Tensordot_1/MatMul/pfor/split:output:0<loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: t
1loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB v
3loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB Ç
+loop_body/Tensordot_1/MatMul/pfor/Reshape_2Reshape0loop_body/Tensordot_1/MatMul/pfor/split:output:1<loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: t
1loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB v
3loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB Ç
+loop_body/Tensordot_1/MatMul/pfor/Reshape_3Reshape0loop_body/Tensordot_1/MatMul/pfor/split:output:2<loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ¹
%loop_body/Tensordot_1/MatMul/pfor/mulMul4loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: È
1loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePack4loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:0)loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:á
+loop_body/Tensordot_1/MatMul/pfor/Reshape_4Reshape2loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0:loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÂ
(loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul&loop_body/Tensordot_1/Reshape:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
3loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
1loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePack<loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:î
+loop_body/Tensordot_1/MatMul/pfor/Reshape_5Reshape2loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0:loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ã
-loop_body/Tensordot_1/MatMul/pfor/transpose_1	Transpose4loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0;loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
&loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : É
!loop_body/Tensordot_1/pfor/concatConcatV2pfor/Reshape:output:0$loop_body/Tensordot_1/shape:output:0/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:¾
"loop_body/Tensordot_1/pfor/ReshapeReshape1loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:0*loop_body/Tensordot_1/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
loop_body/add/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :[
loop_body/add/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :Z
loop_body/add/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :
loop_body/add/pfor/addAddV2"loop_body/add/pfor/Rank_1:output:0!loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: 
loop_body/add/pfor/MaximumMaximumloop_body/add/pfor/add:z:0 loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: s
loop_body/add/pfor/ShapeShape+loop_body/Tensordot_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
loop_body/add/pfor/subSubloop_body/add/pfor/Maximum:z:0 loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: j
 loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
loop_body/add/pfor/ReshapeReshapeloop_body/add/pfor/sub:z:0)loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:g
loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:
loop_body/add/pfor/TileTile&loop_body/add/pfor/Tile/input:output:0#loop_body/add/pfor/Reshape:output:0*
T0*
_output_shapes
: p
&loop_body/add/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(loop_body/add/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(loop_body/add/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
 loop_body/add/pfor/strided_sliceStridedSlice!loop_body/add/pfor/Shape:output:0/loop_body/add/pfor/strided_slice/stack:output:01loop_body/add/pfor/strided_slice/stack_1:output:01loop_body/add/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskr
(loop_body/add/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*loop_body/add/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: t
*loop_body/add/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:´
"loop_body/add/pfor/strided_slice_1StridedSlice!loop_body/add/pfor/Shape:output:01loop_body/add/pfor/strided_slice_1/stack:output:03loop_body/add/pfor/strided_slice_1/stack_1:output:03loop_body/add/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask`
loop_body/add/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ö
loop_body/add/pfor/concatConcatV2)loop_body/add/pfor/strided_slice:output:0 loop_body/add/pfor/Tile:output:0+loop_body/add/pfor/strided_slice_1:output:0'loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ª
loop_body/add/pfor/Reshape_1Reshape+loop_body/Tensordot_1/pfor/Reshape:output:0"loop_body/add/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
loop_body/add/pfor/AddV2AddV2%loop_body/add/pfor/Reshape_1:output:0$loop_body/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
SoftmaxSoftmaxloop_body/add/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^loop_body/ReadVariableOp^loop_body/ReadVariableOp_1^loop_body/add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 24
loop_body/ReadVariableOploop_body/ReadVariableOp28
loop_body/ReadVariableOp_1loop_body/ReadVariableOp_12<
loop_body/add/ReadVariableOploop_body/add/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·û

@__inference_model_layer_call_and_return_conditional_losses_37860

inputs;
)tn__toy_loop_body_readvariableop_resource:A
+tn__toy_loop_body_readvariableop_1_resource:;
-tn__toy_loop_body_add_readvariableop_resource:
identity¢ tn__toy/loop_body/ReadVariableOp¢"tn__toy/loop_body/ReadVariableOp_1¢$tn__toy/loop_body/add/ReadVariableOpC
tn__toy/ShapeShapeinputs*
T0*
_output_shapes
:e
tn__toy/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
tn__toy/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
tn__toy/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
tn__toy/strided_sliceStridedSlicetn__toy/Shape:output:0$tn__toy/strided_slice/stack:output:0&tn__toy/strided_slice/stack_1:output:0&tn__toy/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
tn__toy/Rank/packedPacktn__toy/strided_slice:output:0*
N*
T0*
_output_shapes
:N
tn__toy/RankConst*
_output_shapes
: *
dtype0*
value	B :U
tn__toy/range/startConst*
_output_shapes
: *
dtype0*
value	B : U
tn__toy/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
tn__toy/rangeRangetn__toy/range/start:output:0tn__toy/Rank:output:0tn__toy/range/delta:output:0*
_output_shapes
:g
tn__toy/Max/inputPacktn__toy/strided_slice:output:0*
N*
T0*
_output_shapes
:g
tn__toy/MaxMaxtn__toy/Max/input:output:0tn__toy/range:output:0*
T0*
_output_shapes
: p
.tn__toy/loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : ­
(tn__toy/loop_body/PlaceholderWithDefaultPlaceholderWithDefault7tn__toy/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: M
tn__toy/loop_body/ShapeShapeinputs*
T0*
_output_shapes
:o
%tn__toy/loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'tn__toy/loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'tn__toy/loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
tn__toy/loop_body/strided_sliceStridedSlice tn__toy/loop_body/Shape:output:0.tn__toy/loop_body/strided_slice/stack:output:00tn__toy/loop_body/strided_slice/stack_1:output:00tn__toy/loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
tn__toy/loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :
tn__toy/loop_body/GreaterGreater(tn__toy/loop_body/strided_slice:output:0$tn__toy/loop_body/Greater/y:output:0*
T0*
_output_shapes
: ^
tn__toy/loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : À
tn__toy/loop_body/SelectV2SelectV2tn__toy/loop_body/Greater:z:01tn__toy/loop_body/PlaceholderWithDefault:output:0%tn__toy/loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: a
tn__toy/loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : É
tn__toy/loop_body/GatherV2GatherV2inputs#tn__toy/loop_body/SelectV2:output:0(tn__toy/loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes

:
 tn__toy/loop_body/ReadVariableOpReadVariableOp)tn__toy_loop_body_readvariableop_resource*
_output_shapes

:*
dtype0
"tn__toy/loop_body/ReadVariableOp_1ReadVariableOp+tn__toy_loop_body_readvariableop_1_resource*"
_output_shapes
:*
dtype0{
*tn__toy/loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       µ
%tn__toy/loop_body/Tensordot/transpose	Transpose#tn__toy/loop_body/GatherV2:output:03tn__toy/loop_body/Tensordot/transpose/perm:output:0*
T0*
_output_shapes

:ª
"tn__toy/loop_body/Tensordot/MatMulMatMul)tn__toy/loop_body/Tensordot/transpose:y:0(tn__toy/loop_body/ReadVariableOp:value:0*
T0*
_output_shapes

:
,tn__toy/loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ä
'tn__toy/loop_body/Tensordot_1/transpose	Transpose*tn__toy/loop_body/ReadVariableOp_1:value:05tn__toy/loop_body/Tensordot_1/transpose/perm:output:0*
T0*"
_output_shapes
:|
+tn__toy/loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ¼
%tn__toy/loop_body/Tensordot_1/ReshapeReshape+tn__toy/loop_body/Tensordot_1/transpose:y:04tn__toy/loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:~
-tn__toy/loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Á
'tn__toy/loop_body/Tensordot_1/Reshape_1Reshape,tn__toy/loop_body/Tensordot/MatMul:product:06tn__toy/loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:¹
$tn__toy/loop_body/Tensordot_1/MatMulMatMul.tn__toy/loop_body/Tensordot_1/Reshape:output:00tn__toy/loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes

:m
#tn__toy/loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:«
tn__toy/loop_body/Tensordot_1Reshape.tn__toy/loop_body/Tensordot_1/MatMul:product:0,tn__toy/loop_body/Tensordot_1/shape:output:0*
T0*
_output_shapes
:
$tn__toy/loop_body/add/ReadVariableOpReadVariableOp-tn__toy_loop_body_add_readvariableop_resource*
_output_shapes
:*
dtype0
tn__toy/loop_body/addAddV2&tn__toy/loop_body/Tensordot_1:output:0,tn__toy/loop_body/add/ReadVariableOp:value:0*
T0*
_output_shapes
:d
tn__toy/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
tn__toy/pfor/ReshapeReshapetn__toy/Max:output:0#tn__toy/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:Z
tn__toy/pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : Z
tn__toy/pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
tn__toy/pfor/rangeRange!tn__toy/pfor/range/start:output:0tn__toy/Max:output:0!tn__toy/pfor/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$tn__toy/loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : g
%tn__toy/loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :¬
#tn__toy/loop_body/SelectV2/pfor/addAddV2-tn__toy/loop_body/SelectV2/pfor/Rank:output:0.tn__toy/loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: h
&tn__toy/loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :h
&tn__toy/loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : i
'tn__toy/loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :²
%tn__toy/loop_body/SelectV2/pfor/add_1AddV2/tn__toy/loop_body/SelectV2/pfor/Rank_2:output:00tn__toy/loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: ­
'tn__toy/loop_body/SelectV2/pfor/MaximumMaximum/tn__toy/loop_body/SelectV2/pfor/Rank_1:output:0'tn__toy/loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: ­
)tn__toy/loop_body/SelectV2/pfor/Maximum_1Maximum)tn__toy/loop_body/SelectV2/pfor/add_1:z:0+tn__toy/loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: p
%tn__toy/loop_body/SelectV2/pfor/ShapeShapetn__toy/pfor/range:output:0*
T0*
_output_shapes
:«
#tn__toy/loop_body/SelectV2/pfor/subSub-tn__toy/loop_body/SelectV2/pfor/Maximum_1:z:0/tn__toy/loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: w
-tn__toy/loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:¸
'tn__toy/loop_body/SelectV2/pfor/ReshapeReshape'tn__toy/loop_body/SelectV2/pfor/sub:z:06tn__toy/loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:t
*tn__toy/loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:¶
$tn__toy/loop_body/SelectV2/pfor/TileTile3tn__toy/loop_body/SelectV2/pfor/Tile/input:output:00tn__toy/loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: }
3tn__toy/loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tn__toy/loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tn__toy/loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
-tn__toy/loop_body/SelectV2/pfor/strided_sliceStridedSlice.tn__toy/loop_body/SelectV2/pfor/Shape:output:0<tn__toy/loop_body/SelectV2/pfor/strided_slice/stack:output:0>tn__toy/loop_body/SelectV2/pfor/strided_slice/stack_1:output:0>tn__toy/loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
5tn__toy/loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
7tn__toy/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
7tn__toy/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ó
/tn__toy/loop_body/SelectV2/pfor/strided_slice_1StridedSlice.tn__toy/loop_body/SelectV2/pfor/Shape:output:0>tn__toy/loop_body/SelectV2/pfor/strided_slice_1/stack:output:0@tn__toy/loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:0@tn__toy/loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskm
+tn__toy/loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
&tn__toy/loop_body/SelectV2/pfor/concatConcatV26tn__toy/loop_body/SelectV2/pfor/strided_slice:output:0-tn__toy/loop_body/SelectV2/pfor/Tile:output:08tn__toy/loop_body/SelectV2/pfor/strided_slice_1:output:04tn__toy/loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:°
)tn__toy/loop_body/SelectV2/pfor/Reshape_1Reshapetn__toy/pfor/range:output:0/tn__toy/loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
(tn__toy/loop_body/SelectV2/pfor/SelectV2SelectV2tn__toy/loop_body/Greater:z:02tn__toy/loop_body/SelectV2/pfor/Reshape_1:output:0%tn__toy/loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
-tn__toy/loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
(tn__toy/loop_body/GatherV2/pfor/GatherV2GatherV2inputs1tn__toy/loop_body/SelectV2/pfor/SelectV2:output:06tn__toy/loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0tn__toy/loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ì
.tn__toy/loop_body/Tensordot/transpose/pfor/addAddV23tn__toy/loop_body/Tensordot/transpose/perm:output:09tn__toy/loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:
:tn__toy/loop_body/Tensordot/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: x
6tn__toy/loop_body/Tensordot/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¥
1tn__toy/loop_body/Tensordot/transpose/pfor/concatConcatV2Ctn__toy/loop_body/Tensordot/transpose/pfor/concat/values_0:output:02tn__toy/loop_body/Tensordot/transpose/pfor/add:z:0?tn__toy/loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:æ
4tn__toy/loop_body/Tensordot/transpose/pfor/Transpose	Transpose1tn__toy/loop_body/GatherV2/pfor/GatherV2:output:0:tn__toy/loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-tn__toy/loop_body/Tensordot/MatMul/pfor/ShapeShape8tn__toy/loop_body/Tensordot/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:y
7tn__toy/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ò
-tn__toy/loop_body/Tensordot/MatMul/pfor/splitSplit@tn__toy/loop_body/Tensordot/MatMul/pfor/split/split_dim:output:06tn__toy/loop_body/Tensordot/MatMul/pfor/Shape:output:0*
T0*&
_output_shapes
:::*
	num_splitx
5tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB z
7tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB Õ
/tn__toy/loop_body/Tensordot/MatMul/pfor/ReshapeReshape6tn__toy/loop_body/Tensordot/MatMul/pfor/split:output:0@tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1:output:0*
T0*
_output_shapes
: z
7tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB |
9tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB Ù
1tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_1Reshape6tn__toy/loop_body/Tensordot/MatMul/pfor/split:output:1Btn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: z
7tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB |
9tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB Ù
1tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_2Reshape6tn__toy/loop_body/Tensordot/MatMul/pfor/split:output:2Btn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: É
+tn__toy/loop_body/Tensordot/MatMul/pfor/mulMul8tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape:output:0:tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: Ú
7tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack/tn__toy/loop_body/Tensordot/MatMul/pfor/mul:z:0:tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:ó
1tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape8tn__toy/loop_body/Tensordot/transpose/pfor/Transpose:y:0@tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÐ
.tn__toy/loop_body/Tensordot/MatMul/pfor/MatMulMatMul:tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0(tn__toy/loop_body/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
9tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ§
7tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePack8tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape:output:0:tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0Btn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:î
1tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_4Reshape8tn__toy/loop_body/Tensordot/MatMul/pfor/MatMul:product:0@tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
8tn__toy/loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
3tn__toy/loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2tn__toy/pfor/Reshape:output:06tn__toy/loop_body/Tensordot_1/Reshape_1/shape:output:0Atn__toy/loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ï
4tn__toy/loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape:tn__toy/loop_body/Tensordot/MatMul/pfor/Reshape_4:output:0<tn__toy/loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/tn__toy/loop_body/Tensordot_1/MatMul/pfor/ShapeShape=tn__toy/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
=tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:£
7tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSlice8tn__toy/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Ftn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0Htn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0Htn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Atn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Atn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
9tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSlice8tn__toy/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Htn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Jtn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Jtn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskã
1tn__toy/loop_body/Tensordot_1/MatMul/pfor/MinimumMinimum@tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0Btn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: s
1tn__toy/loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :Ì
/tn__toy/loop_body/Tensordot_1/MatMul/pfor/EqualEqual5tn__toy/loop_body/Tensordot_1/MatMul/pfor/Minimum:z:0:tn__toy/loop_body/Tensordot_1/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: 
2tn__toy/loop_body/Tensordot_1/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          
2tn__toy/loop_body/Tensordot_1/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          
0tn__toy/loop_body/Tensordot_1/MatMul/pfor/SelectSelect3tn__toy/loop_body/Tensordot_1/MatMul/pfor/Equal:z:0;tn__toy/loop_body/Tensordot_1/MatMul/pfor/Select/t:output:0;tn__toy/loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:
?tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
Atn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Atn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
9tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSlice8tn__toy/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Htn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Jtn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Jtn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Atn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Atn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
9tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSlice8tn__toy/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Htn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Jtn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Jtn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
Atn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Atn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
9tn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSlice8tn__toy/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Htn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Jtn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Jtn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask±
/tn__toy/loop_body/Tensordot_1/MatMul/pfor/stackPackBtn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0Btn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0Btn__toy/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:
3tn__toy/loop_body/Tensordot_1/MatMul/pfor/transpose	Transpose=tn__toy/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:09tn__toy/loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿå
1tn__toy/loop_body/Tensordot_1/MatMul/pfor/ReshapeReshape7tn__toy/loop_body/Tensordot_1/MatMul/pfor/transpose:y:08tn__toy/loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1tn__toy/loop_body/Tensordot_1/MatMul/pfor/Shape_1Shape:tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:{
9tn__toy/loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ú
/tn__toy/loop_body/Tensordot_1/MatMul/pfor/splitSplitBtn__toy/loop_body/Tensordot_1/MatMul/pfor/split/split_dim:output:0:tn__toy/loop_body/Tensordot_1/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split|
9tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ~
;tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ß
3tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_1Reshape8tn__toy/loop_body/Tensordot_1/MatMul/pfor/split:output:0Dtn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: |
9tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ~
;tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ß
3tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_2Reshape8tn__toy/loop_body/Tensordot_1/MatMul/pfor/split:output:1Dtn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: |
9tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB ~
;tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB ß
3tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_3Reshape8tn__toy/loop_body/Tensordot_1/MatMul/pfor/split:output:2Dtn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: Ñ
-tn__toy/loop_body/Tensordot_1/MatMul/pfor/mulMul<tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0<tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: à
9tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePack<tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:01tn__toy/loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:ù
3tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_4Reshape:tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0Btn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
0tn__toy/loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul.tn__toy/loop_body/Tensordot_1/Reshape:output:0<tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
;tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ±
9tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePackDtn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:0<tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0<tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:
3tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_5Reshape:tn__toy/loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0Btn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:tn__toy/loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          û
5tn__toy/loop_body/Tensordot_1/MatMul/pfor/transpose_1	Transpose<tn__toy/loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0Ctn__toy/loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
.tn__toy/loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : é
)tn__toy/loop_body/Tensordot_1/pfor/concatConcatV2tn__toy/pfor/Reshape:output:0,tn__toy/loop_body/Tensordot_1/shape:output:07tn__toy/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ö
*tn__toy/loop_body/Tensordot_1/pfor/ReshapeReshape9tn__toy/loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:02tn__toy/loop_body/Tensordot_1/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
tn__toy/loop_body/add/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :c
!tn__toy/loop_body/add/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :b
 tn__toy/loop_body/add/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :
tn__toy/loop_body/add/pfor/addAddV2*tn__toy/loop_body/add/pfor/Rank_1:output:0)tn__toy/loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: 
"tn__toy/loop_body/add/pfor/MaximumMaximum"tn__toy/loop_body/add/pfor/add:z:0(tn__toy/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: 
 tn__toy/loop_body/add/pfor/ShapeShape3tn__toy/loop_body/Tensordot_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
tn__toy/loop_body/add/pfor/subSub&tn__toy/loop_body/add/pfor/Maximum:z:0(tn__toy/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: r
(tn__toy/loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:©
"tn__toy/loop_body/add/pfor/ReshapeReshape"tn__toy/loop_body/add/pfor/sub:z:01tn__toy/loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:o
%tn__toy/loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:§
tn__toy/loop_body/add/pfor/TileTile.tn__toy/loop_body/add/pfor/Tile/input:output:0+tn__toy/loop_body/add/pfor/Reshape:output:0*
T0*
_output_shapes
: x
.tn__toy/loop_body/add/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0tn__toy/loop_body/add/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0tn__toy/loop_body/add/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ö
(tn__toy/loop_body/add/pfor/strided_sliceStridedSlice)tn__toy/loop_body/add/pfor/Shape:output:07tn__toy/loop_body/add/pfor/strided_slice/stack:output:09tn__toy/loop_body/add/pfor/strided_slice/stack_1:output:09tn__toy/loop_body/add/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskz
0tn__toy/loop_body/add/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:|
2tn__toy/loop_body/add/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2tn__toy/loop_body/add/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ü
*tn__toy/loop_body/add/pfor/strided_slice_1StridedSlice)tn__toy/loop_body/add/pfor/Shape:output:09tn__toy/loop_body/add/pfor/strided_slice_1/stack:output:0;tn__toy/loop_body/add/pfor/strided_slice_1/stack_1:output:0;tn__toy/loop_body/add/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskh
&tn__toy/loop_body/add/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
!tn__toy/loop_body/add/pfor/concatConcatV21tn__toy/loop_body/add/pfor/strided_slice:output:0(tn__toy/loop_body/add/pfor/Tile:output:03tn__toy/loop_body/add/pfor/strided_slice_1:output:0/tn__toy/loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Â
$tn__toy/loop_body/add/pfor/Reshape_1Reshape3tn__toy/loop_body/Tensordot_1/pfor/Reshape:output:0*tn__toy/loop_body/add/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
 tn__toy/loop_body/add/pfor/AddV2AddV2-tn__toy/loop_body/add/pfor/Reshape_1:output:0,tn__toy/loop_body/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
tn__toy/SoftmaxSoftmax$tn__toy/loop_body/add/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitytn__toy/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
NoOpNoOp!^tn__toy/loop_body/ReadVariableOp#^tn__toy/loop_body/ReadVariableOp_1%^tn__toy/loop_body/add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2D
 tn__toy/loop_body/ReadVariableOp tn__toy/loop_body/ReadVariableOp2H
"tn__toy/loop_body/ReadVariableOp_1"tn__toy/loop_body/ReadVariableOp_12L
$tn__toy/loop_body/add/ReadVariableOp$tn__toy/loop_body/add/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø
´
%__inference_model_layer_call_fn_37606
input_1
unknown:
	unknown_0:
	unknown_1:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_37586o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ø
´
%__inference_model_layer_call_fn_37550
input_1
unknown:
	unknown_0:
	unknown_1:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_37541o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
õ
³
%__inference_model_layer_call_fn_37655

inputs
unknown:
	unknown_0:
	unknown_1:
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_37541o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*®
serving_default
?
input_14
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ;
tn__toy0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:0

layer-0
layer_with_weights-0
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
Ã
mps1
mps2
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
I
iter
	decay
learning_rate
momentum"
	optimizer
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
â2ß
%__inference_model_layer_call_fn_37550
%__inference_model_layer_call_fn_37655
%__inference_model_layer_call_fn_37666
%__inference_model_layer_call_fn_37606À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Î2Ë
@__inference_model_layer_call_and_return_conditional_losses_37860
@__inference_model_layer_call_and_return_conditional_losses_38054
@__inference_model_layer_call_and_return_conditional_losses_37617
@__inference_model_layer_call_and_return_conditional_losses_37628À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ËBÈ
 __inference__wrapped_model_37333input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
serving_default"
signature_map
:2mps1
:2mps2
:2bias
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
non_trainable_variables

 layers
!metrics
"layer_regularization_losses
#layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_tn__toy_layer_call_fn_38084¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_tn__toy_layer_call_and_return_conditional_losses_38276¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÊBÇ
#__inference_signature_wrapper_38067input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	&total
	'count
(	variables
)	keras_api"
_tf_keras_metric
q
*
thresholds
+true_positives
,false_positives
-	variables
.	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
&0
'1"
trackable_list_wrapper
-
(	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
.
+0
,1"
trackable_list_wrapper
-
-	variables"
_generic_user_object
 __inference__wrapped_model_37333n4¢1
*¢'
%"
input_1ÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
tn__toy!
tn__toyÿÿÿÿÿÿÿÿÿ®
@__inference_model_layer_call_and_return_conditional_losses_37617j<¢9
2¢/
%"
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ®
@__inference_model_layer_call_and_return_conditional_losses_37628j<¢9
2¢/
%"
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ­
@__inference_model_layer_call_and_return_conditional_losses_37860i;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ­
@__inference_model_layer_call_and_return_conditional_losses_38054i;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
%__inference_model_layer_call_fn_37550]<¢9
2¢/
%"
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
%__inference_model_layer_call_fn_37606]<¢9
2¢/
%"
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
%__inference_model_layer_call_fn_37655\;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
%__inference_model_layer_call_fn_37666\;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ 
#__inference_signature_wrapper_38067y?¢<
¢ 
5ª2
0
input_1%"
input_1ÿÿÿÿÿÿÿÿÿ"1ª.
,
tn__toy!
tn__toyÿÿÿÿÿÿÿÿÿ§
B__inference_tn__toy_layer_call_and_return_conditional_losses_38276a3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
'__inference_tn__toy_layer_call_fn_38084T3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ