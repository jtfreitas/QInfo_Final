??
?#?#
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
?
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
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
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
delete_old_dirsbool(?
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
2	?
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
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
7
Square
x"T
y"T"
Ttype:
2	
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
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
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??
|
conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1/kernel
u
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*&
_output_shapes
:*
dtype0
l

conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
conv1/bias
e
conv1/bias/Read/ReadVariableOpReadVariableOp
conv1/bias*
_output_shapes
:*
dtype0
b
aVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namea
[
a/Read/ReadVariableOpReadVariableOpa*"
_output_shapes
:*
dtype0
b
bVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameb
[
b/Read/ReadVariableOpReadVariableOpb*"
_output_shapes
:?*
dtype0
d
biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namebias
]
bias/Read/ReadVariableOpReadVariableOpbias*
_output_shapes

:*
dtype0
}
out_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*!
shared_nameout_layer/kernel
v
$out_layer/kernel/Read/ReadVariableOpReadVariableOpout_layer/kernel*
_output_shapes
:	?
*
dtype0
t
out_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameout_layer/bias
m
"out_layer/bias/Read/ReadVariableOpReadVariableOpout_layer/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
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
?
Adam/conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/conv1/kernel/m
?
'Adam/conv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1/kernel/m*&
_output_shapes
:*
dtype0
z
Adam/conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/conv1/bias/m
s
%Adam/conv1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1/bias/m*
_output_shapes
:*
dtype0
p
Adam/a/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Adam/a/m
i
Adam/a/m/Read/ReadVariableOpReadVariableOpAdam/a/m*"
_output_shapes
:*
dtype0
p
Adam/b/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
Adam/b/m
i
Adam/b/m/Read/ReadVariableOpReadVariableOpAdam/b/m*"
_output_shapes
:?*
dtype0
r
Adam/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameAdam/bias/m
k
Adam/bias/m/Read/ReadVariableOpReadVariableOpAdam/bias/m*
_output_shapes

:*
dtype0
?
Adam/out_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*(
shared_nameAdam/out_layer/kernel/m
?
+Adam/out_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/out_layer/kernel/m*
_output_shapes
:	?
*
dtype0
?
Adam/out_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/out_layer/bias/m
{
)Adam/out_layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/out_layer/bias/m*
_output_shapes
:
*
dtype0
?
Adam/conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/conv1/kernel/v
?
'Adam/conv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1/kernel/v*&
_output_shapes
:*
dtype0
z
Adam/conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/conv1/bias/v
s
%Adam/conv1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1/bias/v*
_output_shapes
:*
dtype0
p
Adam/a/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Adam/a/v
i
Adam/a/v/Read/ReadVariableOpReadVariableOpAdam/a/v*"
_output_shapes
:*
dtype0
p
Adam/b/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
Adam/b/v
i
Adam/b/v/Read/ReadVariableOpReadVariableOpAdam/b/v*"
_output_shapes
:?*
dtype0
r
Adam/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameAdam/bias/v
k
Adam/bias/v/Read/ReadVariableOpReadVariableOpAdam/bias/v*
_output_shapes

:*
dtype0
?
Adam/out_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*(
shared_nameAdam/out_layer/kernel/v
?
+Adam/out_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/out_layer/kernel/v*
_output_shapes
:	?
*
dtype0
?
Adam/out_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/out_layer/bias/v
{
)Adam/out_layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/out_layer/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
?7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?7
value?7B?7 B?7
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
?
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses* 
?
	#a_var
	$b_var
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses*
?

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses*
?
4iter

5beta_1

6beta_2
	7decay
8learning_ratemgmh#mi$mj%mk,ml-mmvnvo#vp$vq%vr,vs-vt*
5
0
1
#2
$3
%4
,5
-6*
5
0
1
#2
$3
%4
,5
-6*
* 
?
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

>serving_default* 
\V
VARIABLE_VALUEconv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
?
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
Dactivity_regularizer_fn
*&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses* 
* 
* 
PJ
VARIABLE_VALUEa5layer_with_weights-1/a_var/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEb5layer_with_weights-1/b_var/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEbias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

#0
$1
%2*

#0
$1
%2*
* 
?
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
Uactivity_regularizer_fn
*+&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*
* 
* 
`Z
VARIABLE_VALUEout_layer/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEout_layer/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

,0
-1*

,0
-1*
* 
?
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

\0
]1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	^total
	_count
`	variables
a	keras_api*
[
b
thresholds
ctrue_positives
dfalse_positives
e	variables
f	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

^0
_1*

`	variables*
* 
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

c0
d1*

e	variables*
y
VARIABLE_VALUEAdam/conv1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/a/mQlayer_with_weights-1/a_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/b/mQlayer_with_weights-1/b_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/out_layer/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/out_layer/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/a/vQlayer_with_weights-1/a_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/b/vQlayer_with_weights-1/b_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/out_layer/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/out_layer/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_input_11Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_11conv1/kernel
conv1/biasabbiasout_layer/kernelout_layer/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_312305
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOpa/Read/ReadVariableOpb/Read/ReadVariableOpbias/Read/ReadVariableOp$out_layer/kernel/Read/ReadVariableOp"out_layer/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp'Adam/conv1/kernel/m/Read/ReadVariableOp%Adam/conv1/bias/m/Read/ReadVariableOpAdam/a/m/Read/ReadVariableOpAdam/b/m/Read/ReadVariableOpAdam/bias/m/Read/ReadVariableOp+Adam/out_layer/kernel/m/Read/ReadVariableOp)Adam/out_layer/bias/m/Read/ReadVariableOp'Adam/conv1/kernel/v/Read/ReadVariableOp%Adam/conv1/bias/v/Read/ReadVariableOpAdam/a/v/Read/ReadVariableOpAdam/b/v/Read/ReadVariableOpAdam/bias/v/Read/ReadVariableOp+Adam/out_layer/kernel/v/Read/ReadVariableOp)Adam/out_layer/bias/v/Read/ReadVariableOpConst*+
Tin$
"2 	*
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
GPU2*0J 8? *(
f#R!
__inference__traced_save_312737
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1/kernel
conv1/biasabbiasout_layer/kernelout_layer/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttrue_positivesfalse_positivesAdam/conv1/kernel/mAdam/conv1/bias/mAdam/a/mAdam/b/mAdam/bias/mAdam/out_layer/kernel/mAdam/out_layer/bias/mAdam/conv1/kernel/vAdam/conv1/bias/vAdam/a/vAdam/b/vAdam/bias/vAdam/out_layer/kernel/vAdam/out_layer/bias/v**
Tin#
!2*
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
GPU2*0J 8? *+
f&R$
"__inference__traced_restore_312837??
?@
?
__inference__traced_save_312737
file_prefix+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop 
savev2_a_read_readvariableop 
savev2_b_read_readvariableop#
savev2_bias_read_readvariableop/
+savev2_out_layer_kernel_read_readvariableop-
)savev2_out_layer_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop2
.savev2_adam_conv1_kernel_m_read_readvariableop0
,savev2_adam_conv1_bias_m_read_readvariableop'
#savev2_adam_a_m_read_readvariableop'
#savev2_adam_b_m_read_readvariableop*
&savev2_adam_bias_m_read_readvariableop6
2savev2_adam_out_layer_kernel_m_read_readvariableop4
0savev2_adam_out_layer_bias_m_read_readvariableop2
.savev2_adam_conv1_kernel_v_read_readvariableop0
,savev2_adam_conv1_bias_v_read_readvariableop'
#savev2_adam_a_v_read_readvariableop'
#savev2_adam_b_v_read_readvariableop*
&savev2_adam_bias_v_read_readvariableop6
2savev2_adam_out_layer_kernel_v_read_readvariableop4
0savev2_adam_out_layer_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/a_var/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/b_var/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/a_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/b_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/a_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/b_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableopsavev2_a_read_readvariableopsavev2_b_read_readvariableopsavev2_bias_read_readvariableop+savev2_out_layer_kernel_read_readvariableop)savev2_out_layer_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop.savev2_adam_conv1_kernel_m_read_readvariableop,savev2_adam_conv1_bias_m_read_readvariableop#savev2_adam_a_m_read_readvariableop#savev2_adam_b_m_read_readvariableop&savev2_adam_bias_m_read_readvariableop2savev2_adam_out_layer_kernel_m_read_readvariableop0savev2_adam_out_layer_bias_m_read_readvariableop.savev2_adam_conv1_kernel_v_read_readvariableop,savev2_adam_conv1_bias_v_read_readvariableop#savev2_adam_a_v_read_readvariableop#savev2_adam_b_v_read_readvariableop&savev2_adam_bias_v_read_readvariableop2savev2_adam_out_layer_kernel_v_read_readvariableop0savev2_adam_out_layer_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *-
dtypes#
!2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::?::	?
:
: : : : : : : ::::::?::	?
:
::::?::	?
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
::($
"
_output_shapes
:?:$ 

_output_shapes

::%!

_output_shapes
:	?
: 

_output_shapes
:
:

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
::($
"
_output_shapes
:?:$ 

_output_shapes

::%!

_output_shapes
:	?
: 

_output_shapes
:
:,(
&
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
::($
"
_output_shapes
:?:$ 

_output_shapes

::%!

_output_shapes
:	?
: 

_output_shapes
:
:

_output_shapes
: 
?	
?
$__inference_signature_wrapper_312305
input_11!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:?
	unknown_3:
	unknown_4:	?

	unknown_5:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_311099o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_11
?
`
D__inference_maxpool1_layer_call_and_return_conditional_losses_312335

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?1
?
F__inference_sequential_layer_call_and_return_conditional_losses_311578

inputs&
conv1_311540:
conv1_311542:%
mnist_tn_311555:%
mnist_tn_311557:?!
mnist_tn_311559:#
out_layer_311570:	?

out_layer_311572:

identity

identity_1

identity_2??conv1/StatefulPartitionedCall? mnist_tn/StatefulPartitionedCall?!out_layer/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_311540conv1_311542*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_311155?
)conv1/ActivityRegularizer/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU2*0J 8? *6
f1R/
-__inference_conv1_activity_regularizer_311112u
conv1/ActivityRegularizer/ShapeShape&conv1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-conv1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/conv1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/conv1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'conv1/ActivityRegularizer/strided_sliceStridedSlice(conv1/ActivityRegularizer/Shape:output:06conv1/ActivityRegularizer/strided_slice/stack:output:08conv1/ActivityRegularizer/strided_slice/stack_1:output:08conv1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv1/ActivityRegularizer/CastCast0conv1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
!conv1/ActivityRegularizer/truedivRealDiv2conv1/ActivityRegularizer/PartitionedCall:output:0"conv1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
maxpool1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_311121?
flatten/PartitionedCallPartitionedCall!maxpool1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_311176?
 mnist_tn/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0mnist_tn_311555mnist_tn_311557mnist_tn_311559*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_mnist_tn_layer_call_and_return_conditional_losses_311395?
,mnist_tn/ActivityRegularizer/PartitionedCallPartitionedCall)mnist_tn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU2*0J 8? *9
f4R2
0__inference_mnist_tn_activity_regularizer_311137{
"mnist_tn/ActivityRegularizer/ShapeShape)mnist_tn/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0mnist_tn/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2mnist_tn/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2mnist_tn/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*mnist_tn/ActivityRegularizer/strided_sliceStridedSlice+mnist_tn/ActivityRegularizer/Shape:output:09mnist_tn/ActivityRegularizer/strided_slice/stack:output:0;mnist_tn/ActivityRegularizer/strided_slice/stack_1:output:0;mnist_tn/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!mnist_tn/ActivityRegularizer/CastCast3mnist_tn/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$mnist_tn/ActivityRegularizer/truedivRealDiv5mnist_tn/ActivityRegularizer/PartitionedCall:output:0%mnist_tn/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
!out_layer/StatefulPartitionedCallStatefulPartitionedCall)mnist_tn/StatefulPartitionedCall:output:0out_layer_311570out_layer_311572*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_out_layer_layer_call_and_return_conditional_losses_311422y
IdentityIdentity*out_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
e

Identity_1Identity%conv1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_2Identity(mnist_tn/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^conv1/StatefulPartitionedCall!^mnist_tn/StatefulPartitionedCall"^out_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2D
 mnist_tn/StatefulPartitionedCall mnist_tn/StatefulPartitionedCall2F
!out_layer/StatefulPartitionedCall!out_layer/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_maxpool1_layer_call_fn_312330

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_311121?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_mnist_tn_layer_call_fn_312363

inputs
unknown:
	unknown_0:?
	unknown_1:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_mnist_tn_layer_call_and_return_conditional_losses_311395p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
D
-__inference_conv1_activity_regularizer_311112
x
identity6
SquareSquarex*
T0*
_output_shapes
:9
RankRank
Square:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????G
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
??
?
D__inference_mnist_tn_layer_call_and_return_conditional_losses_311395

inputs7
!loop_body_readvariableop_resource:9
#loop_body_readvariableop_1_resource:?7
%loop_body_add_readvariableop_resource:
identity??loop_body/ReadVariableOp?loop_body/ReadVariableOp_1?loop_body/add/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:?
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
value	B : ?
 loop_body/PlaceholderWithDefaultPlaceholderWithDefault/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: E
loop_body/ShapeShapeinputs*
T0*
_output_shapes
:g
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
valueB:?
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
value	B : ?
loop_body/SelectV2SelectV2loop_body/Greater:z:0)loop_body/PlaceholderWithDefault:output:0loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: Y
loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
loop_body/GatherV2GatherV2inputsloop_body/SelectV2:output:0 loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes	
:?h
loop_body/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   ?
loop_body/ReshapeReshapeloop_body/GatherV2:output:0 loop_body/Reshape/shape:output:0*
T0*
_output_shapes

:?~
loop_body/ReadVariableOpReadVariableOp!loop_body_readvariableop_resource*"
_output_shapes
:*
dtype0?
loop_body/ReadVariableOp_1ReadVariableOp#loop_body_readvariableop_1_resource*"
_output_shapes
:?*
dtype0s
"loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ?
loop_body/Tensordot/transpose	Transposeloop_body/Reshape:output:0+loop_body/Tensordot/transpose/perm:output:0*
T0*
_output_shapes

:?y
$loop_body/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
loop_body/Tensordot/transpose_1	Transpose loop_body/ReadVariableOp:value:0-loop_body/Tensordot/transpose_1/perm:output:0*
T0*"
_output_shapes
:r
!loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   ?
loop_body/Tensordot/ReshapeReshape#loop_body/Tensordot/transpose_1:y:0*loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	??
loop_body/Tensordot/MatMulMatMul!loop_body/Tensordot/transpose:y:0$loop_body/Tensordot/Reshape:output:0*
T0*
_output_shapes
:	??n
loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"?         ?
loop_body/TensordotReshape$loop_body/Tensordot/MatMul:product:0"loop_body/Tensordot/shape:output:0*
T0*"
_output_shapes
:?t
#loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?  ?
loop_body/Tensordot_1/ReshapeReshape"loop_body/ReadVariableOp_1:value:0,loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes
:	?y
$loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
loop_body/Tensordot_1/transpose	Transposeloop_body/Tensordot:output:0-loop_body/Tensordot_1/transpose/perm:output:0*
T0*"
_output_shapes
:?v
%loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"?     ?
loop_body/Tensordot_1/Reshape_1Reshape#loop_body/Tensordot_1/transpose:y:0.loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	??
loop_body/Tensordot_1/MatMulMatMul&loop_body/Tensordot_1/Reshape:output:0(loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes

:i
loop_body/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ?
loop_body/transpose	Transpose&loop_body/Tensordot_1/MatMul:product:0!loop_body/transpose/perm:output:0*
T0*
_output_shapes

:?
loop_body/add/ReadVariableOpReadVariableOp%loop_body_add_readvariableop_resource*
_output_shapes

:*
dtype0~
loop_body/addAddV2loop_body/transpose:y:0$loop_body/add/ReadVariableOp:value:0*
T0*
_output_shapes

:\
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
:?????????^
loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : _
loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
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
value	B :?
loop_body/SelectV2/pfor/add_1AddV2'loop_body/SelectV2/pfor/Rank_2:output:0(loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: ?
loop_body/SelectV2/pfor/MaximumMaximum'loop_body/SelectV2/pfor/Rank_1:output:0loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: ?
!loop_body/SelectV2/pfor/Maximum_1Maximum!loop_body/SelectV2/pfor/add_1:z:0#loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: `
loop_body/SelectV2/pfor/ShapeShapepfor/range:output:0*
T0*
_output_shapes
:?
loop_body/SelectV2/pfor/subSub%loop_body/SelectV2/pfor/Maximum_1:z:0'loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: o
%loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
loop_body/SelectV2/pfor/ReshapeReshapeloop_body/SelectV2/pfor/sub:z:0.loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:l
"loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:?
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
valueB:?
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
valueB:?
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
value	B : ?
loop_body/SelectV2/pfor/concatConcatV2.loop_body/SelectV2/pfor/strided_slice:output:0%loop_body/SelectV2/pfor/Tile:output:00loop_body/SelectV2/pfor/strided_slice_1:output:0,loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
!loop_body/SelectV2/pfor/Reshape_1Reshapepfor/range:output:0'loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:??????????
 loop_body/SelectV2/pfor/SelectV2SelectV2loop_body/Greater:z:0*loop_body/SelectV2/pfor/Reshape_1:output:0loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:?????????g
%loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
 loop_body/GatherV2/pfor/GatherV2GatherV2inputs)loop_body/SelectV2/pfor/SelectV2:output:0.loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*(
_output_shapes
:??????????d
"loop_body/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
loop_body/Reshape/pfor/concatConcatV2pfor/Reshape:output:0 loop_body/Reshape/shape:output:0+loop_body/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
loop_body/Reshape/pfor/ReshapeReshape)loop_body/GatherV2/pfor/GatherV2:output:0&loop_body/Reshape/pfor/concat:output:0*
T0*+
_output_shapes
:??????????j
(loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
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
value	B : ?
)loop_body/Tensordot/transpose/pfor/concatConcatV2;loop_body/Tensordot/transpose/pfor/concat/values_0:output:0*loop_body/Tensordot/transpose/pfor/add:z:07loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,loop_body/Tensordot/transpose/pfor/Transpose	Transpose'loop_body/Reshape/pfor/Reshape:output:02loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*+
_output_shapes
:???????????
%loop_body/Tensordot/MatMul/pfor/ShapeShape0loop_body/Tensordot/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:q
/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
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
valueB ?
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
valueB ?
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
valueB ?
)loop_body/Tensordot/MatMul/pfor/Reshape_2Reshape.loop_body/Tensordot/MatMul/pfor/split:output:2:loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ?
#loop_body/Tensordot/MatMul/pfor/mulMul0loop_body/Tensordot/MatMul/pfor/Reshape:output:02loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: ?
/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack'loop_body/Tensordot/MatMul/pfor/mul:z:02loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:?
)loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape0loop_body/Tensordot/transpose/pfor/Transpose:y:08loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:???????????????????
&loop_body/Tensordot/MatMul/pfor/MatMulMatMul2loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0$loop_body/Tensordot/Reshape:output:0*
T0*(
_output_shapes
:??????????|
1loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
??????????
/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePack0loop_body/Tensordot/MatMul/pfor/Reshape:output:02loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0:loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:?
)loop_body/Tensordot/MatMul/pfor/Reshape_4Reshape0loop_body/Tensordot/MatMul/pfor/MatMul:product:08loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*,
_output_shapes
:???????????f
$loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
loop_body/Tensordot/pfor/concatConcatV2pfor/Reshape:output:0"loop_body/Tensordot/shape:output:0-loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
 loop_body/Tensordot/pfor/ReshapeReshape2loop_body/Tensordot/MatMul/pfor/Reshape_4:output:0(loop_body/Tensordot/pfor/concat:output:0*
T0*/
_output_shapes
:??????????l
*loop_body/Tensordot_1/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
(loop_body/Tensordot_1/transpose/pfor/addAddV2-loop_body/Tensordot_1/transpose/perm:output:03loop_body/Tensordot_1/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:~
4loop_body/Tensordot_1/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: r
0loop_body/Tensordot_1/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+loop_body/Tensordot_1/transpose/pfor/concatConcatV2=loop_body/Tensordot_1/transpose/pfor/concat/values_0:output:0,loop_body/Tensordot_1/transpose/pfor/add:z:09loop_body/Tensordot_1/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
.loop_body/Tensordot_1/transpose/pfor/Transpose	Transpose)loop_body/Tensordot/pfor/Reshape:output:04loop_body/Tensordot_1/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:??????????r
0loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2pfor/Reshape:output:0.loop_body/Tensordot_1/Reshape_1/shape:output:09loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape2loop_body/Tensordot_1/transpose/pfor/Transpose:y:04loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:???????????
'loop_body/Tensordot_1/MatMul/pfor/ShapeShape5loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
5loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0>loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
)loop_body/Tensordot_1/MatMul/pfor/MinimumMinimum8loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: k
)loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :?
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
valueB"          ?
(loop_body/Tensordot_1/MatMul/pfor/SelectSelect+loop_body/Tensordot_1/MatMul/pfor/Equal:z:03loop_body/Tensordot_1/MatMul/pfor/Select/t:output:03loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:?
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
'loop_body/Tensordot_1/MatMul/pfor/stackPack:loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
+loop_body/Tensordot_1/MatMul/pfor/transpose	Transpose5loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:01loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'????????????????????????????
)loop_body/Tensordot_1/MatMul/pfor/ReshapeReshape/loop_body/Tensordot_1/MatMul/pfor/transpose:y:00loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:???????????
)loop_body/Tensordot_1/MatMul/pfor/Shape_1Shape2loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:s
1loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
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
valueB ?
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
valueB ?
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
valueB ?
+loop_body/Tensordot_1/MatMul/pfor/Reshape_3Reshape0loop_body/Tensordot_1/MatMul/pfor/split:output:2<loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ?
%loop_body/Tensordot_1/MatMul/pfor/mulMul4loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ?
1loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePack4loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:0)loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:?
+loop_body/Tensordot_1/MatMul/pfor/Reshape_4Reshape2loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0:loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:???????????????????
(loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul&loop_body/Tensordot_1/Reshape:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:?????????~
3loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
??????????
1loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePack<loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:?
+loop_body/Tensordot_1/MatMul/pfor/Reshape_5Reshape2loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0:loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'????????????????????????????
2loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
-loop_body/Tensordot_1/MatMul/pfor/transpose_1	Transpose4loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0;loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????`
loop_body/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
loop_body/transpose/pfor/addAddV2!loop_body/transpose/perm:output:0'loop_body/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:r
(loop_body/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: f
$loop_body/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
loop_body/transpose/pfor/concatConcatV21loop_body/transpose/pfor/concat/values_0:output:0 loop_body/transpose/pfor/add:z:0-loop_body/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
"loop_body/transpose/pfor/Transpose	Transpose1loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:0(loop_body/transpose/pfor/concat:output:0*
T0*+
_output_shapes
:?????????Y
loop_body/add/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :[
loop_body/add/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :Z
loop_body/add/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
loop_body/add/pfor/addAddV2"loop_body/add/pfor/Rank_1:output:0!loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: ?
loop_body/add/pfor/MaximumMaximumloop_body/add/pfor/add:z:0 loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: n
loop_body/add/pfor/ShapeShape&loop_body/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:?
loop_body/add/pfor/subSubloop_body/add/pfor/Maximum:z:0 loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: j
 loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
loop_body/add/pfor/ReshapeReshapeloop_body/add/pfor/sub:z:0)loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:g
loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:?
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
valueB:?
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
valueB:?
"loop_body/add/pfor/strided_slice_1StridedSlice!loop_body/add/pfor/Shape:output:01loop_body/add/pfor/strided_slice_1/stack:output:03loop_body/add/pfor/strided_slice_1/stack_1:output:03loop_body/add/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask`
loop_body/add/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
loop_body/add/pfor/concatConcatV2)loop_body/add/pfor/strided_slice:output:0 loop_body/add/pfor/Tile:output:0+loop_body/add/pfor/strided_slice_1:output:0'loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
loop_body/add/pfor/Reshape_1Reshape&loop_body/transpose/pfor/Transpose:y:0"loop_body/add/pfor/concat:output:0*
T0*+
_output_shapes
:??????????
loop_body/add/pfor/AddV2AddV2%loop_body/add/pfor/Reshape_1:output:0$loop_body/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   {
ReshapeReshapeloop_body/add/pfor/AddV2:z:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????Q
ReluReluReshape:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^loop_body/ReadVariableOp^loop_body/ReadVariableOp_1^loop_body/add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????: : : 24
loop_body/ReadVariableOploop_body/ReadVariableOp28
loop_body/ReadVariableOp_1loop_body/ReadVariableOp_12<
loop_body/add/ReadVariableOploop_body/add/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
+__inference_sequential_layer_call_fn_311450
input_11!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:?
	unknown_3:
	unknown_4:	?

	unknown_5:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
: : *)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_311431o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_11
?w
?
"__inference__traced_restore_312837
file_prefix7
assignvariableop_conv1_kernel:+
assignvariableop_1_conv1_bias:*
assignvariableop_2_a:*
assignvariableop_3_b:?)
assignvariableop_4_bias:6
#assignvariableop_5_out_layer_kernel:	?
/
!assignvariableop_6_out_layer_bias:
&
assignvariableop_7_adam_iter:	 (
assignvariableop_8_adam_beta_1: (
assignvariableop_9_adam_beta_2: (
assignvariableop_10_adam_decay: 0
&assignvariableop_11_adam_learning_rate: #
assignvariableop_12_total: #
assignvariableop_13_count: 0
"assignvariableop_14_true_positives:1
#assignvariableop_15_false_positives:A
'assignvariableop_16_adam_conv1_kernel_m:3
%assignvariableop_17_adam_conv1_bias_m:2
assignvariableop_18_adam_a_m:2
assignvariableop_19_adam_b_m:?1
assignvariableop_20_adam_bias_m:>
+assignvariableop_21_adam_out_layer_kernel_m:	?
7
)assignvariableop_22_adam_out_layer_bias_m:
A
'assignvariableop_23_adam_conv1_kernel_v:3
%assignvariableop_24_adam_conv1_bias_v:2
assignvariableop_25_adam_a_v:2
assignvariableop_26_adam_b_v:?1
assignvariableop_27_adam_bias_v:>
+assignvariableop_28_adam_out_layer_kernel_v:	?
7
)assignvariableop_29_adam_out_layer_bias_v:

identity_31??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/a_var/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/b_var/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/a_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/b_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/a_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/b_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_aIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_bIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_out_layer_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_out_layer_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_iterIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_2Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_decayIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp&assignvariableop_11_adam_learning_rateIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_true_positivesIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp#assignvariableop_15_false_positivesIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_conv1_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp%assignvariableop_17_adam_conv1_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_a_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_b_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_out_layer_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_out_layer_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_conv1_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_conv1_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_a_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_b_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp+assignvariableop_28_adam_out_layer_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_out_layer_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_31Identity_31:output:0*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
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
??
?
F__inference_sequential_layer_call_and_return_conditional_losses_312022

inputs>
$conv1_conv2d_readvariableop_resource:3
%conv1_biasadd_readvariableop_resource:@
*mnist_tn_loop_body_readvariableop_resource:B
,mnist_tn_loop_body_readvariableop_1_resource:?@
.mnist_tn_loop_body_add_readvariableop_resource:;
(out_layer_matmul_readvariableop_resource:	?
7
)out_layer_biasadd_readvariableop_resource:

identity

identity_1

identity_2??conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?!mnist_tn/loop_body/ReadVariableOp?#mnist_tn/loop_body/ReadVariableOp_1?%mnist_tn/loop_body/add/ReadVariableOp? out_layer/BiasAdd/ReadVariableOp?out_layer/MatMul/ReadVariableOp?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv1/Conv2DConv2Dinputs#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
~
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????~
 conv1/ActivityRegularizer/SquareSquareconv1/Relu:activations:0*
T0*/
_output_shapes
:?????????x
conv1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
conv1/ActivityRegularizer/SumSum$conv1/ActivityRegularizer/Square:y:0(conv1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: d
conv1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
conv1/ActivityRegularizer/mulMul(conv1/ActivityRegularizer/mul/x:output:0&conv1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: g
conv1/ActivityRegularizer/ShapeShapeconv1/Relu:activations:0*
T0*
_output_shapes
:w
-conv1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/conv1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/conv1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'conv1/ActivityRegularizer/strided_sliceStridedSlice(conv1/ActivityRegularizer/Shape:output:06conv1/ActivityRegularizer/strided_slice/stack:output:08conv1/ActivityRegularizer/strided_slice/stack_1:output:08conv1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv1/ActivityRegularizer/CastCast0conv1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
!conv1/ActivityRegularizer/truedivRealDiv!conv1/ActivityRegularizer/mul:z:0"conv1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
maxpool1/MaxPoolMaxPoolconv1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
flatten/ReshapeReshapemaxpool1/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????V
mnist_tn/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:f
mnist_tn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
mnist_tn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
mnist_tn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
mnist_tn/strided_sliceStridedSlicemnist_tn/Shape:output:0%mnist_tn/strided_slice/stack:output:0'mnist_tn/strided_slice/stack_1:output:0'mnist_tn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
mnist_tn/Rank/packedPackmnist_tn/strided_slice:output:0*
N*
T0*
_output_shapes
:O
mnist_tn/RankConst*
_output_shapes
: *
dtype0*
value	B :V
mnist_tn/range/startConst*
_output_shapes
: *
dtype0*
value	B : V
mnist_tn/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
mnist_tn/rangeRangemnist_tn/range/start:output:0mnist_tn/Rank:output:0mnist_tn/range/delta:output:0*
_output_shapes
:i
mnist_tn/Max/inputPackmnist_tn/strided_slice:output:0*
N*
T0*
_output_shapes
:j
mnist_tn/MaxMaxmnist_tn/Max/input:output:0mnist_tn/range:output:0*
T0*
_output_shapes
: q
/mnist_tn/loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : ?
)mnist_tn/loop_body/PlaceholderWithDefaultPlaceholderWithDefault8mnist_tn/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: `
mnist_tn/loop_body/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:p
&mnist_tn/loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(mnist_tn/loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(mnist_tn/loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 mnist_tn/loop_body/strided_sliceStridedSlice!mnist_tn/loop_body/Shape:output:0/mnist_tn/loop_body/strided_slice/stack:output:01mnist_tn/loop_body/strided_slice/stack_1:output:01mnist_tn/loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
mnist_tn/loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :?
mnist_tn/loop_body/GreaterGreater)mnist_tn/loop_body/strided_slice:output:0%mnist_tn/loop_body/Greater/y:output:0*
T0*
_output_shapes
: _
mnist_tn/loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : ?
mnist_tn/loop_body/SelectV2SelectV2mnist_tn/loop_body/Greater:z:02mnist_tn/loop_body/PlaceholderWithDefault:output:0&mnist_tn/loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: b
 mnist_tn/loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
mnist_tn/loop_body/GatherV2GatherV2flatten/Reshape:output:0$mnist_tn/loop_body/SelectV2:output:0)mnist_tn/loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes	
:?q
 mnist_tn/loop_body/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   ?
mnist_tn/loop_body/ReshapeReshape$mnist_tn/loop_body/GatherV2:output:0)mnist_tn/loop_body/Reshape/shape:output:0*
T0*
_output_shapes

:??
!mnist_tn/loop_body/ReadVariableOpReadVariableOp*mnist_tn_loop_body_readvariableop_resource*"
_output_shapes
:*
dtype0?
#mnist_tn/loop_body/ReadVariableOp_1ReadVariableOp,mnist_tn_loop_body_readvariableop_1_resource*"
_output_shapes
:?*
dtype0|
+mnist_tn/loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ?
&mnist_tn/loop_body/Tensordot/transpose	Transpose#mnist_tn/loop_body/Reshape:output:04mnist_tn/loop_body/Tensordot/transpose/perm:output:0*
T0*
_output_shapes

:??
-mnist_tn/loop_body/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
(mnist_tn/loop_body/Tensordot/transpose_1	Transpose)mnist_tn/loop_body/ReadVariableOp:value:06mnist_tn/loop_body/Tensordot/transpose_1/perm:output:0*
T0*"
_output_shapes
:{
*mnist_tn/loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   ?
$mnist_tn/loop_body/Tensordot/ReshapeReshape,mnist_tn/loop_body/Tensordot/transpose_1:y:03mnist_tn/loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	??
#mnist_tn/loop_body/Tensordot/MatMulMatMul*mnist_tn/loop_body/Tensordot/transpose:y:0-mnist_tn/loop_body/Tensordot/Reshape:output:0*
T0*
_output_shapes
:	??w
"mnist_tn/loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"?         ?
mnist_tn/loop_body/TensordotReshape-mnist_tn/loop_body/Tensordot/MatMul:product:0+mnist_tn/loop_body/Tensordot/shape:output:0*
T0*"
_output_shapes
:?}
,mnist_tn/loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?  ?
&mnist_tn/loop_body/Tensordot_1/ReshapeReshape+mnist_tn/loop_body/ReadVariableOp_1:value:05mnist_tn/loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes
:	??
-mnist_tn/loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
(mnist_tn/loop_body/Tensordot_1/transpose	Transpose%mnist_tn/loop_body/Tensordot:output:06mnist_tn/loop_body/Tensordot_1/transpose/perm:output:0*
T0*"
_output_shapes
:?
.mnist_tn/loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"?     ?
(mnist_tn/loop_body/Tensordot_1/Reshape_1Reshape,mnist_tn/loop_body/Tensordot_1/transpose:y:07mnist_tn/loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	??
%mnist_tn/loop_body/Tensordot_1/MatMulMatMul/mnist_tn/loop_body/Tensordot_1/Reshape:output:01mnist_tn/loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes

:r
!mnist_tn/loop_body/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ?
mnist_tn/loop_body/transpose	Transpose/mnist_tn/loop_body/Tensordot_1/MatMul:product:0*mnist_tn/loop_body/transpose/perm:output:0*
T0*
_output_shapes

:?
%mnist_tn/loop_body/add/ReadVariableOpReadVariableOp.mnist_tn_loop_body_add_readvariableop_resource*
_output_shapes

:*
dtype0?
mnist_tn/loop_body/addAddV2 mnist_tn/loop_body/transpose:y:0-mnist_tn/loop_body/add/ReadVariableOp:value:0*
T0*
_output_shapes

:e
mnist_tn/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
mnist_tn/pfor/ReshapeReshapemnist_tn/Max:output:0$mnist_tn/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:[
mnist_tn/pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : [
mnist_tn/pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
mnist_tn/pfor/rangeRange"mnist_tn/pfor/range/start:output:0mnist_tn/Max:output:0"mnist_tn/pfor/range/delta:output:0*#
_output_shapes
:?????????g
%mnist_tn/loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : h
&mnist_tn/loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
$mnist_tn/loop_body/SelectV2/pfor/addAddV2.mnist_tn/loop_body/SelectV2/pfor/Rank:output:0/mnist_tn/loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: i
'mnist_tn/loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :i
'mnist_tn/loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : j
(mnist_tn/loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
&mnist_tn/loop_body/SelectV2/pfor/add_1AddV20mnist_tn/loop_body/SelectV2/pfor/Rank_2:output:01mnist_tn/loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: ?
(mnist_tn/loop_body/SelectV2/pfor/MaximumMaximum0mnist_tn/loop_body/SelectV2/pfor/Rank_1:output:0(mnist_tn/loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: ?
*mnist_tn/loop_body/SelectV2/pfor/Maximum_1Maximum*mnist_tn/loop_body/SelectV2/pfor/add_1:z:0,mnist_tn/loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: r
&mnist_tn/loop_body/SelectV2/pfor/ShapeShapemnist_tn/pfor/range:output:0*
T0*
_output_shapes
:?
$mnist_tn/loop_body/SelectV2/pfor/subSub.mnist_tn/loop_body/SelectV2/pfor/Maximum_1:z:00mnist_tn/loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: x
.mnist_tn/loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
(mnist_tn/loop_body/SelectV2/pfor/ReshapeReshape(mnist_tn/loop_body/SelectV2/pfor/sub:z:07mnist_tn/loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:u
+mnist_tn/loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:?
%mnist_tn/loop_body/SelectV2/pfor/TileTile4mnist_tn/loop_body/SelectV2/pfor/Tile/input:output:01mnist_tn/loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: ~
4mnist_tn/loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6mnist_tn/loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mnist_tn/loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mnist_tn/loop_body/SelectV2/pfor/strided_sliceStridedSlice/mnist_tn/loop_body/SelectV2/pfor/Shape:output:0=mnist_tn/loop_body/SelectV2/pfor/strided_slice/stack:output:0?mnist_tn/loop_body/SelectV2/pfor/strided_slice/stack_1:output:0?mnist_tn/loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
6mnist_tn/loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
8mnist_tn/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
8mnist_tn/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0mnist_tn/loop_body/SelectV2/pfor/strided_slice_1StridedSlice/mnist_tn/loop_body/SelectV2/pfor/Shape:output:0?mnist_tn/loop_body/SelectV2/pfor/strided_slice_1/stack:output:0Amnist_tn/loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:0Amnist_tn/loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskn
,mnist_tn/loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'mnist_tn/loop_body/SelectV2/pfor/concatConcatV27mnist_tn/loop_body/SelectV2/pfor/strided_slice:output:0.mnist_tn/loop_body/SelectV2/pfor/Tile:output:09mnist_tn/loop_body/SelectV2/pfor/strided_slice_1:output:05mnist_tn/loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
*mnist_tn/loop_body/SelectV2/pfor/Reshape_1Reshapemnist_tn/pfor/range:output:00mnist_tn/loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:??????????
)mnist_tn/loop_body/SelectV2/pfor/SelectV2SelectV2mnist_tn/loop_body/Greater:z:03mnist_tn/loop_body/SelectV2/pfor/Reshape_1:output:0&mnist_tn/loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:?????????p
.mnist_tn/loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)mnist_tn/loop_body/GatherV2/pfor/GatherV2GatherV2flatten/Reshape:output:02mnist_tn/loop_body/SelectV2/pfor/SelectV2:output:07mnist_tn/loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*(
_output_shapes
:??????????m
+mnist_tn/loop_body/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&mnist_tn/loop_body/Reshape/pfor/concatConcatV2mnist_tn/pfor/Reshape:output:0)mnist_tn/loop_body/Reshape/shape:output:04mnist_tn/loop_body/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
'mnist_tn/loop_body/Reshape/pfor/ReshapeReshape2mnist_tn/loop_body/GatherV2/pfor/GatherV2:output:0/mnist_tn/loop_body/Reshape/pfor/concat:output:0*
T0*+
_output_shapes
:??????????s
1mnist_tn/loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
/mnist_tn/loop_body/Tensordot/transpose/pfor/addAddV24mnist_tn/loop_body/Tensordot/transpose/perm:output:0:mnist_tn/loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:?
;mnist_tn/loop_body/Tensordot/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: y
7mnist_tn/loop_body/Tensordot/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2mnist_tn/loop_body/Tensordot/transpose/pfor/concatConcatV2Dmnist_tn/loop_body/Tensordot/transpose/pfor/concat/values_0:output:03mnist_tn/loop_body/Tensordot/transpose/pfor/add:z:0@mnist_tn/loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
5mnist_tn/loop_body/Tensordot/transpose/pfor/Transpose	Transpose0mnist_tn/loop_body/Reshape/pfor/Reshape:output:0;mnist_tn/loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*+
_output_shapes
:???????????
.mnist_tn/loop_body/Tensordot/MatMul/pfor/ShapeShape9mnist_tn/loop_body/Tensordot/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:z
8mnist_tn/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
.mnist_tn/loop_body/Tensordot/MatMul/pfor/splitSplitAmnist_tn/loop_body/Tensordot/MatMul/pfor/split/split_dim:output:07mnist_tn/loop_body/Tensordot/MatMul/pfor/Shape:output:0*
T0*&
_output_shapes
:::*
	num_splity
6mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB {
8mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
0mnist_tn/loop_body/Tensordot/MatMul/pfor/ReshapeReshape7mnist_tn/loop_body/Tensordot/MatMul/pfor/split:output:0Amnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1:output:0*
T0*
_output_shapes
: {
8mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB }
:mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
2mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_1Reshape7mnist_tn/loop_body/Tensordot/MatMul/pfor/split:output:1Cmnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: {
8mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB }
:mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
2mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_2Reshape7mnist_tn/loop_body/Tensordot/MatMul/pfor/split:output:2Cmnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ?
,mnist_tn/loop_body/Tensordot/MatMul/pfor/mulMul9mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape:output:0;mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: ?
8mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack0mnist_tn/loop_body/Tensordot/MatMul/pfor/mul:z:0;mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:?
2mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape9mnist_tn/loop_body/Tensordot/transpose/pfor/Transpose:y:0Amnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:???????????????????
/mnist_tn/loop_body/Tensordot/MatMul/pfor/MatMulMatMul;mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0-mnist_tn/loop_body/Tensordot/Reshape:output:0*
T0*(
_output_shapes
:???????????
:mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
??????????
8mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePack9mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape:output:0;mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0Cmnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:?
2mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_4Reshape9mnist_tn/loop_body/Tensordot/MatMul/pfor/MatMul:product:0Amnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*,
_output_shapes
:???????????o
-mnist_tn/loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(mnist_tn/loop_body/Tensordot/pfor/concatConcatV2mnist_tn/pfor/Reshape:output:0+mnist_tn/loop_body/Tensordot/shape:output:06mnist_tn/loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
)mnist_tn/loop_body/Tensordot/pfor/ReshapeReshape;mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_4:output:01mnist_tn/loop_body/Tensordot/pfor/concat:output:0*
T0*/
_output_shapes
:??????????u
3mnist_tn/loop_body/Tensordot_1/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
1mnist_tn/loop_body/Tensordot_1/transpose/pfor/addAddV26mnist_tn/loop_body/Tensordot_1/transpose/perm:output:0<mnist_tn/loop_body/Tensordot_1/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:?
=mnist_tn/loop_body/Tensordot_1/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: {
9mnist_tn/loop_body/Tensordot_1/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4mnist_tn/loop_body/Tensordot_1/transpose/pfor/concatConcatV2Fmnist_tn/loop_body/Tensordot_1/transpose/pfor/concat/values_0:output:05mnist_tn/loop_body/Tensordot_1/transpose/pfor/add:z:0Bmnist_tn/loop_body/Tensordot_1/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
7mnist_tn/loop_body/Tensordot_1/transpose/pfor/Transpose	Transpose2mnist_tn/loop_body/Tensordot/pfor/Reshape:output:0=mnist_tn/loop_body/Tensordot_1/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:??????????{
9mnist_tn/loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4mnist_tn/loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2mnist_tn/pfor/Reshape:output:07mnist_tn/loop_body/Tensordot_1/Reshape_1/shape:output:0Bmnist_tn/loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
5mnist_tn/loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape;mnist_tn/loop_body/Tensordot_1/transpose/pfor/Transpose:y:0=mnist_tn/loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:???????????
0mnist_tn/loop_body/Tensordot_1/MatMul/pfor/ShapeShape>mnist_tn/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:?
>mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
@mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSlice9mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Gmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0Imnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0Imnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Bmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSlice9mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Imnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Kmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Kmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2mnist_tn/loop_body/Tensordot_1/MatMul/pfor/MinimumMinimumAmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0Cmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: t
2mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :?
0mnist_tn/loop_body/Tensordot_1/MatMul/pfor/EqualEqual6mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Minimum:z:0;mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: ?
3mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          ?
3mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          ?
1mnist_tn/loop_body/Tensordot_1/MatMul/pfor/SelectSelect4mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Equal:z:0<mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Select/t:output:0<mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:?
@mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Bmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSlice9mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Imnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Kmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Kmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Bmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSlice9mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Imnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Kmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Kmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Bmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSlice9mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Imnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Kmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Kmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0mnist_tn/loop_body/Tensordot_1/MatMul/pfor/stackPackCmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0Cmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0Cmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
4mnist_tn/loop_body/Tensordot_1/MatMul/pfor/transpose	Transpose>mnist_tn/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'????????????????????????????
2mnist_tn/loop_body/Tensordot_1/MatMul/pfor/ReshapeReshape8mnist_tn/loop_body/Tensordot_1/MatMul/pfor/transpose:y:09mnist_tn/loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:???????????
2mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Shape_1Shape;mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:|
:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
0mnist_tn/loop_body/Tensordot_1/MatMul/pfor/splitSplitCmnist_tn/loop_body/Tensordot_1/MatMul/pfor/split/split_dim:output:0;mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split}
:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB 
<mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
4mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_1Reshape9mnist_tn/loop_body/Tensordot_1/MatMul/pfor/split:output:0Emnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: }
:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB 
<mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
4mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_2Reshape9mnist_tn/loop_body/Tensordot_1/MatMul/pfor/split:output:1Emnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: }
:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB 
<mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
4mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_3Reshape9mnist_tn/loop_body/Tensordot_1/MatMul/pfor/split:output:2Emnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ?
.mnist_tn/loop_body/Tensordot_1/MatMul/pfor/mulMul=mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0=mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ?
:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePack=mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:02mnist_tn/loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:?
4mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_4Reshape;mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0Cmnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:???????????????????
1mnist_tn/loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul/mnist_tn/loop_body/Tensordot_1/Reshape:output:0=mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:??????????
<mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
??????????
:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePackEmnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:0=mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0=mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:?
4mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_5Reshape;mnist_tn/loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0Cmnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'????????????????????????????
;mnist_tn/loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
6mnist_tn/loop_body/Tensordot_1/MatMul/pfor/transpose_1	Transpose=mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0Dmnist_tn/loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????i
'mnist_tn/loop_body/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
%mnist_tn/loop_body/transpose/pfor/addAddV2*mnist_tn/loop_body/transpose/perm:output:00mnist_tn/loop_body/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:{
1mnist_tn/loop_body/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: o
-mnist_tn/loop_body/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(mnist_tn/loop_body/transpose/pfor/concatConcatV2:mnist_tn/loop_body/transpose/pfor/concat/values_0:output:0)mnist_tn/loop_body/transpose/pfor/add:z:06mnist_tn/loop_body/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
+mnist_tn/loop_body/transpose/pfor/Transpose	Transpose:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:01mnist_tn/loop_body/transpose/pfor/concat:output:0*
T0*+
_output_shapes
:?????????b
 mnist_tn/loop_body/add/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :d
"mnist_tn/loop_body/add/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :c
!mnist_tn/loop_body/add/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
mnist_tn/loop_body/add/pfor/addAddV2+mnist_tn/loop_body/add/pfor/Rank_1:output:0*mnist_tn/loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: ?
#mnist_tn/loop_body/add/pfor/MaximumMaximum#mnist_tn/loop_body/add/pfor/add:z:0)mnist_tn/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: ?
!mnist_tn/loop_body/add/pfor/ShapeShape/mnist_tn/loop_body/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:?
mnist_tn/loop_body/add/pfor/subSub'mnist_tn/loop_body/add/pfor/Maximum:z:0)mnist_tn/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: s
)mnist_tn/loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
#mnist_tn/loop_body/add/pfor/ReshapeReshape#mnist_tn/loop_body/add/pfor/sub:z:02mnist_tn/loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:p
&mnist_tn/loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:?
 mnist_tn/loop_body/add/pfor/TileTile/mnist_tn/loop_body/add/pfor/Tile/input:output:0,mnist_tn/loop_body/add/pfor/Reshape:output:0*
T0*
_output_shapes
: y
/mnist_tn/loop_body/add/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1mnist_tn/loop_body/add/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1mnist_tn/loop_body/add/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)mnist_tn/loop_body/add/pfor/strided_sliceStridedSlice*mnist_tn/loop_body/add/pfor/Shape:output:08mnist_tn/loop_body/add/pfor/strided_slice/stack:output:0:mnist_tn/loop_body/add/pfor/strided_slice/stack_1:output:0:mnist_tn/loop_body/add/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask{
1mnist_tn/loop_body/add/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3mnist_tn/loop_body/add/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3mnist_tn/loop_body/add/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+mnist_tn/loop_body/add/pfor/strided_slice_1StridedSlice*mnist_tn/loop_body/add/pfor/Shape:output:0:mnist_tn/loop_body/add/pfor/strided_slice_1/stack:output:0<mnist_tn/loop_body/add/pfor/strided_slice_1/stack_1:output:0<mnist_tn/loop_body/add/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maski
'mnist_tn/loop_body/add/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
"mnist_tn/loop_body/add/pfor/concatConcatV22mnist_tn/loop_body/add/pfor/strided_slice:output:0)mnist_tn/loop_body/add/pfor/Tile:output:04mnist_tn/loop_body/add/pfor/strided_slice_1:output:00mnist_tn/loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%mnist_tn/loop_body/add/pfor/Reshape_1Reshape/mnist_tn/loop_body/transpose/pfor/Transpose:y:0+mnist_tn/loop_body/add/pfor/concat:output:0*
T0*+
_output_shapes
:??????????
!mnist_tn/loop_body/add/pfor/AddV2AddV2.mnist_tn/loop_body/add/pfor/Reshape_1:output:0-mnist_tn/loop_body/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????g
mnist_tn/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
mnist_tn/ReshapeReshape%mnist_tn/loop_body/add/pfor/AddV2:z:0mnist_tn/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????c
mnist_tn/ReluRelumnist_tn/Reshape:output:0*
T0*(
_output_shapes
:??????????}
#mnist_tn/ActivityRegularizer/SquareSquaremnist_tn/Relu:activations:0*
T0*(
_output_shapes
:??????????s
"mnist_tn/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 mnist_tn/ActivityRegularizer/SumSum'mnist_tn/ActivityRegularizer/Square:y:0+mnist_tn/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: g
"mnist_tn/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
 mnist_tn/ActivityRegularizer/mulMul+mnist_tn/ActivityRegularizer/mul/x:output:0)mnist_tn/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: m
"mnist_tn/ActivityRegularizer/ShapeShapemnist_tn/Relu:activations:0*
T0*
_output_shapes
:z
0mnist_tn/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2mnist_tn/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2mnist_tn/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*mnist_tn/ActivityRegularizer/strided_sliceStridedSlice+mnist_tn/ActivityRegularizer/Shape:output:09mnist_tn/ActivityRegularizer/strided_slice/stack:output:0;mnist_tn/ActivityRegularizer/strided_slice/stack_1:output:0;mnist_tn/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!mnist_tn/ActivityRegularizer/CastCast3mnist_tn/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$mnist_tn/ActivityRegularizer/truedivRealDiv$mnist_tn/ActivityRegularizer/mul:z:0%mnist_tn/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
out_layer/MatMul/ReadVariableOpReadVariableOp(out_layer_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
out_layer/MatMulMatMulmnist_tn/Relu:activations:0'out_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
 out_layer/BiasAdd/ReadVariableOpReadVariableOp)out_layer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
out_layer/BiasAddBiasAddout_layer/MatMul:product:0(out_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
j
out_layer/SoftmaxSoftmaxout_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
j
IdentityIdentityout_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
e

Identity_1Identity%conv1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_2Identity(mnist_tn/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp"^mnist_tn/loop_body/ReadVariableOp$^mnist_tn/loop_body/ReadVariableOp_1&^mnist_tn/loop_body/add/ReadVariableOp!^out_layer/BiasAdd/ReadVariableOp ^out_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : 2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2F
!mnist_tn/loop_body/ReadVariableOp!mnist_tn/loop_body/ReadVariableOp2J
#mnist_tn/loop_body/ReadVariableOp_1#mnist_tn/loop_body/ReadVariableOp_12N
%mnist_tn/loop_body/add/ReadVariableOp%mnist_tn/loop_body/add/ReadVariableOp2D
 out_layer/BiasAdd/ReadVariableOp out_layer/BiasAdd/ReadVariableOp2B
out_layer/MatMul/ReadVariableOpout_layer/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
+__inference_sequential_layer_call_fn_311739

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:?
	unknown_3:
	unknown_4:	?

	unknown_5:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
: : *)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_311431o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
D
(__inference_flatten_layer_call_fn_312340

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_311176a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
D__inference_mnist_tn_layer_call_and_return_conditional_losses_312624

inputs7
!loop_body_readvariableop_resource:9
#loop_body_readvariableop_1_resource:?7
%loop_body_add_readvariableop_resource:
identity??loop_body/ReadVariableOp?loop_body/ReadVariableOp_1?loop_body/add/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:?
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
value	B : ?
 loop_body/PlaceholderWithDefaultPlaceholderWithDefault/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: E
loop_body/ShapeShapeinputs*
T0*
_output_shapes
:g
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
valueB:?
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
value	B : ?
loop_body/SelectV2SelectV2loop_body/Greater:z:0)loop_body/PlaceholderWithDefault:output:0loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: Y
loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
loop_body/GatherV2GatherV2inputsloop_body/SelectV2:output:0 loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes	
:?h
loop_body/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   ?
loop_body/ReshapeReshapeloop_body/GatherV2:output:0 loop_body/Reshape/shape:output:0*
T0*
_output_shapes

:?~
loop_body/ReadVariableOpReadVariableOp!loop_body_readvariableop_resource*"
_output_shapes
:*
dtype0?
loop_body/ReadVariableOp_1ReadVariableOp#loop_body_readvariableop_1_resource*"
_output_shapes
:?*
dtype0s
"loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ?
loop_body/Tensordot/transpose	Transposeloop_body/Reshape:output:0+loop_body/Tensordot/transpose/perm:output:0*
T0*
_output_shapes

:?y
$loop_body/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
loop_body/Tensordot/transpose_1	Transpose loop_body/ReadVariableOp:value:0-loop_body/Tensordot/transpose_1/perm:output:0*
T0*"
_output_shapes
:r
!loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   ?
loop_body/Tensordot/ReshapeReshape#loop_body/Tensordot/transpose_1:y:0*loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	??
loop_body/Tensordot/MatMulMatMul!loop_body/Tensordot/transpose:y:0$loop_body/Tensordot/Reshape:output:0*
T0*
_output_shapes
:	??n
loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"?         ?
loop_body/TensordotReshape$loop_body/Tensordot/MatMul:product:0"loop_body/Tensordot/shape:output:0*
T0*"
_output_shapes
:?t
#loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?  ?
loop_body/Tensordot_1/ReshapeReshape"loop_body/ReadVariableOp_1:value:0,loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes
:	?y
$loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
loop_body/Tensordot_1/transpose	Transposeloop_body/Tensordot:output:0-loop_body/Tensordot_1/transpose/perm:output:0*
T0*"
_output_shapes
:?v
%loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"?     ?
loop_body/Tensordot_1/Reshape_1Reshape#loop_body/Tensordot_1/transpose:y:0.loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	??
loop_body/Tensordot_1/MatMulMatMul&loop_body/Tensordot_1/Reshape:output:0(loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes

:i
loop_body/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ?
loop_body/transpose	Transpose&loop_body/Tensordot_1/MatMul:product:0!loop_body/transpose/perm:output:0*
T0*
_output_shapes

:?
loop_body/add/ReadVariableOpReadVariableOp%loop_body_add_readvariableop_resource*
_output_shapes

:*
dtype0~
loop_body/addAddV2loop_body/transpose:y:0$loop_body/add/ReadVariableOp:value:0*
T0*
_output_shapes

:\
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
:?????????^
loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : _
loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
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
value	B :?
loop_body/SelectV2/pfor/add_1AddV2'loop_body/SelectV2/pfor/Rank_2:output:0(loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: ?
loop_body/SelectV2/pfor/MaximumMaximum'loop_body/SelectV2/pfor/Rank_1:output:0loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: ?
!loop_body/SelectV2/pfor/Maximum_1Maximum!loop_body/SelectV2/pfor/add_1:z:0#loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: `
loop_body/SelectV2/pfor/ShapeShapepfor/range:output:0*
T0*
_output_shapes
:?
loop_body/SelectV2/pfor/subSub%loop_body/SelectV2/pfor/Maximum_1:z:0'loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: o
%loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
loop_body/SelectV2/pfor/ReshapeReshapeloop_body/SelectV2/pfor/sub:z:0.loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:l
"loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:?
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
valueB:?
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
valueB:?
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
value	B : ?
loop_body/SelectV2/pfor/concatConcatV2.loop_body/SelectV2/pfor/strided_slice:output:0%loop_body/SelectV2/pfor/Tile:output:00loop_body/SelectV2/pfor/strided_slice_1:output:0,loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
!loop_body/SelectV2/pfor/Reshape_1Reshapepfor/range:output:0'loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:??????????
 loop_body/SelectV2/pfor/SelectV2SelectV2loop_body/Greater:z:0*loop_body/SelectV2/pfor/Reshape_1:output:0loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:?????????g
%loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
 loop_body/GatherV2/pfor/GatherV2GatherV2inputs)loop_body/SelectV2/pfor/SelectV2:output:0.loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*(
_output_shapes
:??????????d
"loop_body/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
loop_body/Reshape/pfor/concatConcatV2pfor/Reshape:output:0 loop_body/Reshape/shape:output:0+loop_body/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
loop_body/Reshape/pfor/ReshapeReshape)loop_body/GatherV2/pfor/GatherV2:output:0&loop_body/Reshape/pfor/concat:output:0*
T0*+
_output_shapes
:??????????j
(loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
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
value	B : ?
)loop_body/Tensordot/transpose/pfor/concatConcatV2;loop_body/Tensordot/transpose/pfor/concat/values_0:output:0*loop_body/Tensordot/transpose/pfor/add:z:07loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,loop_body/Tensordot/transpose/pfor/Transpose	Transpose'loop_body/Reshape/pfor/Reshape:output:02loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*+
_output_shapes
:???????????
%loop_body/Tensordot/MatMul/pfor/ShapeShape0loop_body/Tensordot/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:q
/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
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
valueB ?
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
valueB ?
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
valueB ?
)loop_body/Tensordot/MatMul/pfor/Reshape_2Reshape.loop_body/Tensordot/MatMul/pfor/split:output:2:loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ?
#loop_body/Tensordot/MatMul/pfor/mulMul0loop_body/Tensordot/MatMul/pfor/Reshape:output:02loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: ?
/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack'loop_body/Tensordot/MatMul/pfor/mul:z:02loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:?
)loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape0loop_body/Tensordot/transpose/pfor/Transpose:y:08loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:???????????????????
&loop_body/Tensordot/MatMul/pfor/MatMulMatMul2loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0$loop_body/Tensordot/Reshape:output:0*
T0*(
_output_shapes
:??????????|
1loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
??????????
/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePack0loop_body/Tensordot/MatMul/pfor/Reshape:output:02loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0:loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:?
)loop_body/Tensordot/MatMul/pfor/Reshape_4Reshape0loop_body/Tensordot/MatMul/pfor/MatMul:product:08loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*,
_output_shapes
:???????????f
$loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
loop_body/Tensordot/pfor/concatConcatV2pfor/Reshape:output:0"loop_body/Tensordot/shape:output:0-loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
 loop_body/Tensordot/pfor/ReshapeReshape2loop_body/Tensordot/MatMul/pfor/Reshape_4:output:0(loop_body/Tensordot/pfor/concat:output:0*
T0*/
_output_shapes
:??????????l
*loop_body/Tensordot_1/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
(loop_body/Tensordot_1/transpose/pfor/addAddV2-loop_body/Tensordot_1/transpose/perm:output:03loop_body/Tensordot_1/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:~
4loop_body/Tensordot_1/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: r
0loop_body/Tensordot_1/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+loop_body/Tensordot_1/transpose/pfor/concatConcatV2=loop_body/Tensordot_1/transpose/pfor/concat/values_0:output:0,loop_body/Tensordot_1/transpose/pfor/add:z:09loop_body/Tensordot_1/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
.loop_body/Tensordot_1/transpose/pfor/Transpose	Transpose)loop_body/Tensordot/pfor/Reshape:output:04loop_body/Tensordot_1/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:??????????r
0loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2pfor/Reshape:output:0.loop_body/Tensordot_1/Reshape_1/shape:output:09loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape2loop_body/Tensordot_1/transpose/pfor/Transpose:y:04loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:???????????
'loop_body/Tensordot_1/MatMul/pfor/ShapeShape5loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
5loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0>loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
)loop_body/Tensordot_1/MatMul/pfor/MinimumMinimum8loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: k
)loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :?
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
valueB"          ?
(loop_body/Tensordot_1/MatMul/pfor/SelectSelect+loop_body/Tensordot_1/MatMul/pfor/Equal:z:03loop_body/Tensordot_1/MatMul/pfor/Select/t:output:03loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:?
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
'loop_body/Tensordot_1/MatMul/pfor/stackPack:loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
+loop_body/Tensordot_1/MatMul/pfor/transpose	Transpose5loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:01loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'????????????????????????????
)loop_body/Tensordot_1/MatMul/pfor/ReshapeReshape/loop_body/Tensordot_1/MatMul/pfor/transpose:y:00loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:???????????
)loop_body/Tensordot_1/MatMul/pfor/Shape_1Shape2loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:s
1loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
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
valueB ?
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
valueB ?
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
valueB ?
+loop_body/Tensordot_1/MatMul/pfor/Reshape_3Reshape0loop_body/Tensordot_1/MatMul/pfor/split:output:2<loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ?
%loop_body/Tensordot_1/MatMul/pfor/mulMul4loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ?
1loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePack4loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:0)loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:?
+loop_body/Tensordot_1/MatMul/pfor/Reshape_4Reshape2loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0:loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:???????????????????
(loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul&loop_body/Tensordot_1/Reshape:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:?????????~
3loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
??????????
1loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePack<loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:?
+loop_body/Tensordot_1/MatMul/pfor/Reshape_5Reshape2loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0:loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'????????????????????????????
2loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
-loop_body/Tensordot_1/MatMul/pfor/transpose_1	Transpose4loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0;loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????`
loop_body/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
loop_body/transpose/pfor/addAddV2!loop_body/transpose/perm:output:0'loop_body/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:r
(loop_body/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: f
$loop_body/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
loop_body/transpose/pfor/concatConcatV21loop_body/transpose/pfor/concat/values_0:output:0 loop_body/transpose/pfor/add:z:0-loop_body/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
"loop_body/transpose/pfor/Transpose	Transpose1loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:0(loop_body/transpose/pfor/concat:output:0*
T0*+
_output_shapes
:?????????Y
loop_body/add/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :[
loop_body/add/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :Z
loop_body/add/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
loop_body/add/pfor/addAddV2"loop_body/add/pfor/Rank_1:output:0!loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: ?
loop_body/add/pfor/MaximumMaximumloop_body/add/pfor/add:z:0 loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: n
loop_body/add/pfor/ShapeShape&loop_body/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:?
loop_body/add/pfor/subSubloop_body/add/pfor/Maximum:z:0 loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: j
 loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
loop_body/add/pfor/ReshapeReshapeloop_body/add/pfor/sub:z:0)loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:g
loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:?
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
valueB:?
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
valueB:?
"loop_body/add/pfor/strided_slice_1StridedSlice!loop_body/add/pfor/Shape:output:01loop_body/add/pfor/strided_slice_1/stack:output:03loop_body/add/pfor/strided_slice_1/stack_1:output:03loop_body/add/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask`
loop_body/add/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
loop_body/add/pfor/concatConcatV2)loop_body/add/pfor/strided_slice:output:0 loop_body/add/pfor/Tile:output:0+loop_body/add/pfor/strided_slice_1:output:0'loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
loop_body/add/pfor/Reshape_1Reshape&loop_body/transpose/pfor/Transpose:y:0"loop_body/add/pfor/concat:output:0*
T0*+
_output_shapes
:??????????
loop_body/add/pfor/AddV2AddV2%loop_body/add/pfor/Reshape_1:output:0$loop_body/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   {
ReshapeReshapeloop_body/add/pfor/AddV2:z:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????Q
ReluReluReshape:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^loop_body/ReadVariableOp^loop_body/ReadVariableOp_1^loop_body/add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????: : : 24
loop_body/ReadVariableOploop_body/ReadVariableOp28
loop_body/ReadVariableOp_1loop_body/ReadVariableOp_12<
loop_body/add/ReadVariableOploop_body/add/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv1_layer_call_and_return_all_conditional_losses_312325

inputs!
unknown:
	unknown_0:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_311155?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU2*0J 8? *6
f1R/
-__inference_conv1_activity_regularizer_311112w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????X

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?1
?
F__inference_sequential_layer_call_and_return_conditional_losses_311431

inputs&
conv1_311156:
conv1_311158:%
mnist_tn_311396:%
mnist_tn_311398:?!
mnist_tn_311400:#
out_layer_311423:	?

out_layer_311425:

identity

identity_1

identity_2??conv1/StatefulPartitionedCall? mnist_tn/StatefulPartitionedCall?!out_layer/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_311156conv1_311158*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_311155?
)conv1/ActivityRegularizer/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU2*0J 8? *6
f1R/
-__inference_conv1_activity_regularizer_311112u
conv1/ActivityRegularizer/ShapeShape&conv1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-conv1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/conv1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/conv1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'conv1/ActivityRegularizer/strided_sliceStridedSlice(conv1/ActivityRegularizer/Shape:output:06conv1/ActivityRegularizer/strided_slice/stack:output:08conv1/ActivityRegularizer/strided_slice/stack_1:output:08conv1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv1/ActivityRegularizer/CastCast0conv1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
!conv1/ActivityRegularizer/truedivRealDiv2conv1/ActivityRegularizer/PartitionedCall:output:0"conv1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
maxpool1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_311121?
flatten/PartitionedCallPartitionedCall!maxpool1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_311176?
 mnist_tn/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0mnist_tn_311396mnist_tn_311398mnist_tn_311400*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_mnist_tn_layer_call_and_return_conditional_losses_311395?
,mnist_tn/ActivityRegularizer/PartitionedCallPartitionedCall)mnist_tn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU2*0J 8? *9
f4R2
0__inference_mnist_tn_activity_regularizer_311137{
"mnist_tn/ActivityRegularizer/ShapeShape)mnist_tn/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0mnist_tn/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2mnist_tn/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2mnist_tn/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*mnist_tn/ActivityRegularizer/strided_sliceStridedSlice+mnist_tn/ActivityRegularizer/Shape:output:09mnist_tn/ActivityRegularizer/strided_slice/stack:output:0;mnist_tn/ActivityRegularizer/strided_slice/stack_1:output:0;mnist_tn/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!mnist_tn/ActivityRegularizer/CastCast3mnist_tn/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$mnist_tn/ActivityRegularizer/truedivRealDiv5mnist_tn/ActivityRegularizer/PartitionedCall:output:0%mnist_tn/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
!out_layer/StatefulPartitionedCallStatefulPartitionedCall)mnist_tn/StatefulPartitionedCall:output:0out_layer_311423out_layer_311425*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_out_layer_layer_call_and_return_conditional_losses_311422y
IdentityIdentity*out_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
e

Identity_1Identity%conv1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_2Identity(mnist_tn/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^conv1/StatefulPartitionedCall!^mnist_tn/StatefulPartitionedCall"^out_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2D
 mnist_tn/StatefulPartitionedCall mnist_tn/StatefulPartitionedCall2F
!out_layer/StatefulPartitionedCall!out_layer/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
F__inference_sequential_layer_call_and_return_conditional_losses_312284

inputs>
$conv1_conv2d_readvariableop_resource:3
%conv1_biasadd_readvariableop_resource:@
*mnist_tn_loop_body_readvariableop_resource:B
,mnist_tn_loop_body_readvariableop_1_resource:?@
.mnist_tn_loop_body_add_readvariableop_resource:;
(out_layer_matmul_readvariableop_resource:	?
7
)out_layer_biasadd_readvariableop_resource:

identity

identity_1

identity_2??conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?!mnist_tn/loop_body/ReadVariableOp?#mnist_tn/loop_body/ReadVariableOp_1?%mnist_tn/loop_body/add/ReadVariableOp? out_layer/BiasAdd/ReadVariableOp?out_layer/MatMul/ReadVariableOp?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv1/Conv2DConv2Dinputs#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
~
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????~
 conv1/ActivityRegularizer/SquareSquareconv1/Relu:activations:0*
T0*/
_output_shapes
:?????????x
conv1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
conv1/ActivityRegularizer/SumSum$conv1/ActivityRegularizer/Square:y:0(conv1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: d
conv1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
conv1/ActivityRegularizer/mulMul(conv1/ActivityRegularizer/mul/x:output:0&conv1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: g
conv1/ActivityRegularizer/ShapeShapeconv1/Relu:activations:0*
T0*
_output_shapes
:w
-conv1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/conv1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/conv1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'conv1/ActivityRegularizer/strided_sliceStridedSlice(conv1/ActivityRegularizer/Shape:output:06conv1/ActivityRegularizer/strided_slice/stack:output:08conv1/ActivityRegularizer/strided_slice/stack_1:output:08conv1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv1/ActivityRegularizer/CastCast0conv1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
!conv1/ActivityRegularizer/truedivRealDiv!conv1/ActivityRegularizer/mul:z:0"conv1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
maxpool1/MaxPoolMaxPoolconv1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
flatten/ReshapeReshapemaxpool1/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????V
mnist_tn/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:f
mnist_tn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
mnist_tn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
mnist_tn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
mnist_tn/strided_sliceStridedSlicemnist_tn/Shape:output:0%mnist_tn/strided_slice/stack:output:0'mnist_tn/strided_slice/stack_1:output:0'mnist_tn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
mnist_tn/Rank/packedPackmnist_tn/strided_slice:output:0*
N*
T0*
_output_shapes
:O
mnist_tn/RankConst*
_output_shapes
: *
dtype0*
value	B :V
mnist_tn/range/startConst*
_output_shapes
: *
dtype0*
value	B : V
mnist_tn/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
mnist_tn/rangeRangemnist_tn/range/start:output:0mnist_tn/Rank:output:0mnist_tn/range/delta:output:0*
_output_shapes
:i
mnist_tn/Max/inputPackmnist_tn/strided_slice:output:0*
N*
T0*
_output_shapes
:j
mnist_tn/MaxMaxmnist_tn/Max/input:output:0mnist_tn/range:output:0*
T0*
_output_shapes
: q
/mnist_tn/loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : ?
)mnist_tn/loop_body/PlaceholderWithDefaultPlaceholderWithDefault8mnist_tn/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: `
mnist_tn/loop_body/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:p
&mnist_tn/loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(mnist_tn/loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(mnist_tn/loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 mnist_tn/loop_body/strided_sliceStridedSlice!mnist_tn/loop_body/Shape:output:0/mnist_tn/loop_body/strided_slice/stack:output:01mnist_tn/loop_body/strided_slice/stack_1:output:01mnist_tn/loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
mnist_tn/loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :?
mnist_tn/loop_body/GreaterGreater)mnist_tn/loop_body/strided_slice:output:0%mnist_tn/loop_body/Greater/y:output:0*
T0*
_output_shapes
: _
mnist_tn/loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : ?
mnist_tn/loop_body/SelectV2SelectV2mnist_tn/loop_body/Greater:z:02mnist_tn/loop_body/PlaceholderWithDefault:output:0&mnist_tn/loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: b
 mnist_tn/loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
mnist_tn/loop_body/GatherV2GatherV2flatten/Reshape:output:0$mnist_tn/loop_body/SelectV2:output:0)mnist_tn/loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes	
:?q
 mnist_tn/loop_body/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   ?
mnist_tn/loop_body/ReshapeReshape$mnist_tn/loop_body/GatherV2:output:0)mnist_tn/loop_body/Reshape/shape:output:0*
T0*
_output_shapes

:??
!mnist_tn/loop_body/ReadVariableOpReadVariableOp*mnist_tn_loop_body_readvariableop_resource*"
_output_shapes
:*
dtype0?
#mnist_tn/loop_body/ReadVariableOp_1ReadVariableOp,mnist_tn_loop_body_readvariableop_1_resource*"
_output_shapes
:?*
dtype0|
+mnist_tn/loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ?
&mnist_tn/loop_body/Tensordot/transpose	Transpose#mnist_tn/loop_body/Reshape:output:04mnist_tn/loop_body/Tensordot/transpose/perm:output:0*
T0*
_output_shapes

:??
-mnist_tn/loop_body/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
(mnist_tn/loop_body/Tensordot/transpose_1	Transpose)mnist_tn/loop_body/ReadVariableOp:value:06mnist_tn/loop_body/Tensordot/transpose_1/perm:output:0*
T0*"
_output_shapes
:{
*mnist_tn/loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   ?
$mnist_tn/loop_body/Tensordot/ReshapeReshape,mnist_tn/loop_body/Tensordot/transpose_1:y:03mnist_tn/loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	??
#mnist_tn/loop_body/Tensordot/MatMulMatMul*mnist_tn/loop_body/Tensordot/transpose:y:0-mnist_tn/loop_body/Tensordot/Reshape:output:0*
T0*
_output_shapes
:	??w
"mnist_tn/loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"?         ?
mnist_tn/loop_body/TensordotReshape-mnist_tn/loop_body/Tensordot/MatMul:product:0+mnist_tn/loop_body/Tensordot/shape:output:0*
T0*"
_output_shapes
:?}
,mnist_tn/loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?  ?
&mnist_tn/loop_body/Tensordot_1/ReshapeReshape+mnist_tn/loop_body/ReadVariableOp_1:value:05mnist_tn/loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes
:	??
-mnist_tn/loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
(mnist_tn/loop_body/Tensordot_1/transpose	Transpose%mnist_tn/loop_body/Tensordot:output:06mnist_tn/loop_body/Tensordot_1/transpose/perm:output:0*
T0*"
_output_shapes
:?
.mnist_tn/loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"?     ?
(mnist_tn/loop_body/Tensordot_1/Reshape_1Reshape,mnist_tn/loop_body/Tensordot_1/transpose:y:07mnist_tn/loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	??
%mnist_tn/loop_body/Tensordot_1/MatMulMatMul/mnist_tn/loop_body/Tensordot_1/Reshape:output:01mnist_tn/loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes

:r
!mnist_tn/loop_body/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ?
mnist_tn/loop_body/transpose	Transpose/mnist_tn/loop_body/Tensordot_1/MatMul:product:0*mnist_tn/loop_body/transpose/perm:output:0*
T0*
_output_shapes

:?
%mnist_tn/loop_body/add/ReadVariableOpReadVariableOp.mnist_tn_loop_body_add_readvariableop_resource*
_output_shapes

:*
dtype0?
mnist_tn/loop_body/addAddV2 mnist_tn/loop_body/transpose:y:0-mnist_tn/loop_body/add/ReadVariableOp:value:0*
T0*
_output_shapes

:e
mnist_tn/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
mnist_tn/pfor/ReshapeReshapemnist_tn/Max:output:0$mnist_tn/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:[
mnist_tn/pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : [
mnist_tn/pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
mnist_tn/pfor/rangeRange"mnist_tn/pfor/range/start:output:0mnist_tn/Max:output:0"mnist_tn/pfor/range/delta:output:0*#
_output_shapes
:?????????g
%mnist_tn/loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : h
&mnist_tn/loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
$mnist_tn/loop_body/SelectV2/pfor/addAddV2.mnist_tn/loop_body/SelectV2/pfor/Rank:output:0/mnist_tn/loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: i
'mnist_tn/loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :i
'mnist_tn/loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : j
(mnist_tn/loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
&mnist_tn/loop_body/SelectV2/pfor/add_1AddV20mnist_tn/loop_body/SelectV2/pfor/Rank_2:output:01mnist_tn/loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: ?
(mnist_tn/loop_body/SelectV2/pfor/MaximumMaximum0mnist_tn/loop_body/SelectV2/pfor/Rank_1:output:0(mnist_tn/loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: ?
*mnist_tn/loop_body/SelectV2/pfor/Maximum_1Maximum*mnist_tn/loop_body/SelectV2/pfor/add_1:z:0,mnist_tn/loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: r
&mnist_tn/loop_body/SelectV2/pfor/ShapeShapemnist_tn/pfor/range:output:0*
T0*
_output_shapes
:?
$mnist_tn/loop_body/SelectV2/pfor/subSub.mnist_tn/loop_body/SelectV2/pfor/Maximum_1:z:00mnist_tn/loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: x
.mnist_tn/loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
(mnist_tn/loop_body/SelectV2/pfor/ReshapeReshape(mnist_tn/loop_body/SelectV2/pfor/sub:z:07mnist_tn/loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:u
+mnist_tn/loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:?
%mnist_tn/loop_body/SelectV2/pfor/TileTile4mnist_tn/loop_body/SelectV2/pfor/Tile/input:output:01mnist_tn/loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: ~
4mnist_tn/loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6mnist_tn/loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6mnist_tn/loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.mnist_tn/loop_body/SelectV2/pfor/strided_sliceStridedSlice/mnist_tn/loop_body/SelectV2/pfor/Shape:output:0=mnist_tn/loop_body/SelectV2/pfor/strided_slice/stack:output:0?mnist_tn/loop_body/SelectV2/pfor/strided_slice/stack_1:output:0?mnist_tn/loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
6mnist_tn/loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
8mnist_tn/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
8mnist_tn/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0mnist_tn/loop_body/SelectV2/pfor/strided_slice_1StridedSlice/mnist_tn/loop_body/SelectV2/pfor/Shape:output:0?mnist_tn/loop_body/SelectV2/pfor/strided_slice_1/stack:output:0Amnist_tn/loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:0Amnist_tn/loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskn
,mnist_tn/loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'mnist_tn/loop_body/SelectV2/pfor/concatConcatV27mnist_tn/loop_body/SelectV2/pfor/strided_slice:output:0.mnist_tn/loop_body/SelectV2/pfor/Tile:output:09mnist_tn/loop_body/SelectV2/pfor/strided_slice_1:output:05mnist_tn/loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
*mnist_tn/loop_body/SelectV2/pfor/Reshape_1Reshapemnist_tn/pfor/range:output:00mnist_tn/loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:??????????
)mnist_tn/loop_body/SelectV2/pfor/SelectV2SelectV2mnist_tn/loop_body/Greater:z:03mnist_tn/loop_body/SelectV2/pfor/Reshape_1:output:0&mnist_tn/loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:?????????p
.mnist_tn/loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)mnist_tn/loop_body/GatherV2/pfor/GatherV2GatherV2flatten/Reshape:output:02mnist_tn/loop_body/SelectV2/pfor/SelectV2:output:07mnist_tn/loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*(
_output_shapes
:??????????m
+mnist_tn/loop_body/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&mnist_tn/loop_body/Reshape/pfor/concatConcatV2mnist_tn/pfor/Reshape:output:0)mnist_tn/loop_body/Reshape/shape:output:04mnist_tn/loop_body/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
'mnist_tn/loop_body/Reshape/pfor/ReshapeReshape2mnist_tn/loop_body/GatherV2/pfor/GatherV2:output:0/mnist_tn/loop_body/Reshape/pfor/concat:output:0*
T0*+
_output_shapes
:??????????s
1mnist_tn/loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
/mnist_tn/loop_body/Tensordot/transpose/pfor/addAddV24mnist_tn/loop_body/Tensordot/transpose/perm:output:0:mnist_tn/loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:?
;mnist_tn/loop_body/Tensordot/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: y
7mnist_tn/loop_body/Tensordot/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2mnist_tn/loop_body/Tensordot/transpose/pfor/concatConcatV2Dmnist_tn/loop_body/Tensordot/transpose/pfor/concat/values_0:output:03mnist_tn/loop_body/Tensordot/transpose/pfor/add:z:0@mnist_tn/loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
5mnist_tn/loop_body/Tensordot/transpose/pfor/Transpose	Transpose0mnist_tn/loop_body/Reshape/pfor/Reshape:output:0;mnist_tn/loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*+
_output_shapes
:???????????
.mnist_tn/loop_body/Tensordot/MatMul/pfor/ShapeShape9mnist_tn/loop_body/Tensordot/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:z
8mnist_tn/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
.mnist_tn/loop_body/Tensordot/MatMul/pfor/splitSplitAmnist_tn/loop_body/Tensordot/MatMul/pfor/split/split_dim:output:07mnist_tn/loop_body/Tensordot/MatMul/pfor/Shape:output:0*
T0*&
_output_shapes
:::*
	num_splity
6mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB {
8mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
0mnist_tn/loop_body/Tensordot/MatMul/pfor/ReshapeReshape7mnist_tn/loop_body/Tensordot/MatMul/pfor/split:output:0Amnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1:output:0*
T0*
_output_shapes
: {
8mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB }
:mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
2mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_1Reshape7mnist_tn/loop_body/Tensordot/MatMul/pfor/split:output:1Cmnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: {
8mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB }
:mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
2mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_2Reshape7mnist_tn/loop_body/Tensordot/MatMul/pfor/split:output:2Cmnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ?
,mnist_tn/loop_body/Tensordot/MatMul/pfor/mulMul9mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape:output:0;mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: ?
8mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack0mnist_tn/loop_body/Tensordot/MatMul/pfor/mul:z:0;mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:?
2mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape9mnist_tn/loop_body/Tensordot/transpose/pfor/Transpose:y:0Amnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:???????????????????
/mnist_tn/loop_body/Tensordot/MatMul/pfor/MatMulMatMul;mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0-mnist_tn/loop_body/Tensordot/Reshape:output:0*
T0*(
_output_shapes
:???????????
:mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
??????????
8mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePack9mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape:output:0;mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0Cmnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:?
2mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_4Reshape9mnist_tn/loop_body/Tensordot/MatMul/pfor/MatMul:product:0Amnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*,
_output_shapes
:???????????o
-mnist_tn/loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(mnist_tn/loop_body/Tensordot/pfor/concatConcatV2mnist_tn/pfor/Reshape:output:0+mnist_tn/loop_body/Tensordot/shape:output:06mnist_tn/loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
)mnist_tn/loop_body/Tensordot/pfor/ReshapeReshape;mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_4:output:01mnist_tn/loop_body/Tensordot/pfor/concat:output:0*
T0*/
_output_shapes
:??????????u
3mnist_tn/loop_body/Tensordot_1/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
1mnist_tn/loop_body/Tensordot_1/transpose/pfor/addAddV26mnist_tn/loop_body/Tensordot_1/transpose/perm:output:0<mnist_tn/loop_body/Tensordot_1/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:?
=mnist_tn/loop_body/Tensordot_1/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: {
9mnist_tn/loop_body/Tensordot_1/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4mnist_tn/loop_body/Tensordot_1/transpose/pfor/concatConcatV2Fmnist_tn/loop_body/Tensordot_1/transpose/pfor/concat/values_0:output:05mnist_tn/loop_body/Tensordot_1/transpose/pfor/add:z:0Bmnist_tn/loop_body/Tensordot_1/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
7mnist_tn/loop_body/Tensordot_1/transpose/pfor/Transpose	Transpose2mnist_tn/loop_body/Tensordot/pfor/Reshape:output:0=mnist_tn/loop_body/Tensordot_1/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:??????????{
9mnist_tn/loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4mnist_tn/loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2mnist_tn/pfor/Reshape:output:07mnist_tn/loop_body/Tensordot_1/Reshape_1/shape:output:0Bmnist_tn/loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
5mnist_tn/loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape;mnist_tn/loop_body/Tensordot_1/transpose/pfor/Transpose:y:0=mnist_tn/loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:???????????
0mnist_tn/loop_body/Tensordot_1/MatMul/pfor/ShapeShape>mnist_tn/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:?
>mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
@mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSlice9mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Gmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0Imnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0Imnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Bmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSlice9mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Imnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Kmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Kmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2mnist_tn/loop_body/Tensordot_1/MatMul/pfor/MinimumMinimumAmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0Cmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: t
2mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :?
0mnist_tn/loop_body/Tensordot_1/MatMul/pfor/EqualEqual6mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Minimum:z:0;mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: ?
3mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          ?
3mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          ?
1mnist_tn/loop_body/Tensordot_1/MatMul/pfor/SelectSelect4mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Equal:z:0<mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Select/t:output:0<mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:?
@mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Bmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSlice9mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Imnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Kmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Kmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Bmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSlice9mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Imnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Kmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Kmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Bmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSlice9mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Imnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Kmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Kmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0mnist_tn/loop_body/Tensordot_1/MatMul/pfor/stackPackCmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0Cmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0Cmnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
4mnist_tn/loop_body/Tensordot_1/MatMul/pfor/transpose	Transpose>mnist_tn/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'????????????????????????????
2mnist_tn/loop_body/Tensordot_1/MatMul/pfor/ReshapeReshape8mnist_tn/loop_body/Tensordot_1/MatMul/pfor/transpose:y:09mnist_tn/loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:???????????
2mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Shape_1Shape;mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:|
:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
0mnist_tn/loop_body/Tensordot_1/MatMul/pfor/splitSplitCmnist_tn/loop_body/Tensordot_1/MatMul/pfor/split/split_dim:output:0;mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split}
:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB 
<mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
4mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_1Reshape9mnist_tn/loop_body/Tensordot_1/MatMul/pfor/split:output:0Emnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: }
:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB 
<mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
4mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_2Reshape9mnist_tn/loop_body/Tensordot_1/MatMul/pfor/split:output:1Emnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: }
:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB 
<mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
4mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_3Reshape9mnist_tn/loop_body/Tensordot_1/MatMul/pfor/split:output:2Emnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ?
.mnist_tn/loop_body/Tensordot_1/MatMul/pfor/mulMul=mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0=mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ?
:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePack=mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:02mnist_tn/loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:?
4mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_4Reshape;mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0Cmnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:???????????????????
1mnist_tn/loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul/mnist_tn/loop_body/Tensordot_1/Reshape:output:0=mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:??????????
<mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
??????????
:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePackEmnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:0=mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0=mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:?
4mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_5Reshape;mnist_tn/loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0Cmnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'????????????????????????????
;mnist_tn/loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
6mnist_tn/loop_body/Tensordot_1/MatMul/pfor/transpose_1	Transpose=mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0Dmnist_tn/loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????i
'mnist_tn/loop_body/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
%mnist_tn/loop_body/transpose/pfor/addAddV2*mnist_tn/loop_body/transpose/perm:output:00mnist_tn/loop_body/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:{
1mnist_tn/loop_body/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: o
-mnist_tn/loop_body/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(mnist_tn/loop_body/transpose/pfor/concatConcatV2:mnist_tn/loop_body/transpose/pfor/concat/values_0:output:0)mnist_tn/loop_body/transpose/pfor/add:z:06mnist_tn/loop_body/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
+mnist_tn/loop_body/transpose/pfor/Transpose	Transpose:mnist_tn/loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:01mnist_tn/loop_body/transpose/pfor/concat:output:0*
T0*+
_output_shapes
:?????????b
 mnist_tn/loop_body/add/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :d
"mnist_tn/loop_body/add/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :c
!mnist_tn/loop_body/add/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
mnist_tn/loop_body/add/pfor/addAddV2+mnist_tn/loop_body/add/pfor/Rank_1:output:0*mnist_tn/loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: ?
#mnist_tn/loop_body/add/pfor/MaximumMaximum#mnist_tn/loop_body/add/pfor/add:z:0)mnist_tn/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: ?
!mnist_tn/loop_body/add/pfor/ShapeShape/mnist_tn/loop_body/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:?
mnist_tn/loop_body/add/pfor/subSub'mnist_tn/loop_body/add/pfor/Maximum:z:0)mnist_tn/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: s
)mnist_tn/loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
#mnist_tn/loop_body/add/pfor/ReshapeReshape#mnist_tn/loop_body/add/pfor/sub:z:02mnist_tn/loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:p
&mnist_tn/loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:?
 mnist_tn/loop_body/add/pfor/TileTile/mnist_tn/loop_body/add/pfor/Tile/input:output:0,mnist_tn/loop_body/add/pfor/Reshape:output:0*
T0*
_output_shapes
: y
/mnist_tn/loop_body/add/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1mnist_tn/loop_body/add/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1mnist_tn/loop_body/add/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)mnist_tn/loop_body/add/pfor/strided_sliceStridedSlice*mnist_tn/loop_body/add/pfor/Shape:output:08mnist_tn/loop_body/add/pfor/strided_slice/stack:output:0:mnist_tn/loop_body/add/pfor/strided_slice/stack_1:output:0:mnist_tn/loop_body/add/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask{
1mnist_tn/loop_body/add/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3mnist_tn/loop_body/add/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3mnist_tn/loop_body/add/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+mnist_tn/loop_body/add/pfor/strided_slice_1StridedSlice*mnist_tn/loop_body/add/pfor/Shape:output:0:mnist_tn/loop_body/add/pfor/strided_slice_1/stack:output:0<mnist_tn/loop_body/add/pfor/strided_slice_1/stack_1:output:0<mnist_tn/loop_body/add/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maski
'mnist_tn/loop_body/add/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
"mnist_tn/loop_body/add/pfor/concatConcatV22mnist_tn/loop_body/add/pfor/strided_slice:output:0)mnist_tn/loop_body/add/pfor/Tile:output:04mnist_tn/loop_body/add/pfor/strided_slice_1:output:00mnist_tn/loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%mnist_tn/loop_body/add/pfor/Reshape_1Reshape/mnist_tn/loop_body/transpose/pfor/Transpose:y:0+mnist_tn/loop_body/add/pfor/concat:output:0*
T0*+
_output_shapes
:??????????
!mnist_tn/loop_body/add/pfor/AddV2AddV2.mnist_tn/loop_body/add/pfor/Reshape_1:output:0-mnist_tn/loop_body/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????g
mnist_tn/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
mnist_tn/ReshapeReshape%mnist_tn/loop_body/add/pfor/AddV2:z:0mnist_tn/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????c
mnist_tn/ReluRelumnist_tn/Reshape:output:0*
T0*(
_output_shapes
:??????????}
#mnist_tn/ActivityRegularizer/SquareSquaremnist_tn/Relu:activations:0*
T0*(
_output_shapes
:??????????s
"mnist_tn/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 mnist_tn/ActivityRegularizer/SumSum'mnist_tn/ActivityRegularizer/Square:y:0+mnist_tn/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: g
"mnist_tn/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
 mnist_tn/ActivityRegularizer/mulMul+mnist_tn/ActivityRegularizer/mul/x:output:0)mnist_tn/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: m
"mnist_tn/ActivityRegularizer/ShapeShapemnist_tn/Relu:activations:0*
T0*
_output_shapes
:z
0mnist_tn/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2mnist_tn/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2mnist_tn/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*mnist_tn/ActivityRegularizer/strided_sliceStridedSlice+mnist_tn/ActivityRegularizer/Shape:output:09mnist_tn/ActivityRegularizer/strided_slice/stack:output:0;mnist_tn/ActivityRegularizer/strided_slice/stack_1:output:0;mnist_tn/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!mnist_tn/ActivityRegularizer/CastCast3mnist_tn/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$mnist_tn/ActivityRegularizer/truedivRealDiv$mnist_tn/ActivityRegularizer/mul:z:0%mnist_tn/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
out_layer/MatMul/ReadVariableOpReadVariableOp(out_layer_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
out_layer/MatMulMatMulmnist_tn/Relu:activations:0'out_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
 out_layer/BiasAdd/ReadVariableOpReadVariableOp)out_layer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
out_layer/BiasAddBiasAddout_layer/MatMul:product:0(out_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
j
out_layer/SoftmaxSoftmaxout_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
j
IdentityIdentityout_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
e

Identity_1Identity%conv1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_2Identity(mnist_tn/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp"^mnist_tn/loop_body/ReadVariableOp$^mnist_tn/loop_body/ReadVariableOp_1&^mnist_tn/loop_body/add/ReadVariableOp!^out_layer/BiasAdd/ReadVariableOp ^out_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : 2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2F
!mnist_tn/loop_body/ReadVariableOp!mnist_tn/loop_body/ReadVariableOp2J
#mnist_tn/loop_body/ReadVariableOp_1#mnist_tn/loop_body/ReadVariableOp_12N
%mnist_tn/loop_body/add/ReadVariableOp%mnist_tn/loop_body/add/ReadVariableOp2D
 out_layer/BiasAdd/ReadVariableOp out_layer/BiasAdd/ReadVariableOp2B
out_layer/MatMul/ReadVariableOpout_layer/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?1
?
F__inference_sequential_layer_call_and_return_conditional_losses_311659
input_11&
conv1_311621:
conv1_311623:%
mnist_tn_311636:%
mnist_tn_311638:?!
mnist_tn_311640:#
out_layer_311651:	?

out_layer_311653:

identity

identity_1

identity_2??conv1/StatefulPartitionedCall? mnist_tn/StatefulPartitionedCall?!out_layer/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallinput_11conv1_311621conv1_311623*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_311155?
)conv1/ActivityRegularizer/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU2*0J 8? *6
f1R/
-__inference_conv1_activity_regularizer_311112u
conv1/ActivityRegularizer/ShapeShape&conv1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-conv1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/conv1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/conv1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'conv1/ActivityRegularizer/strided_sliceStridedSlice(conv1/ActivityRegularizer/Shape:output:06conv1/ActivityRegularizer/strided_slice/stack:output:08conv1/ActivityRegularizer/strided_slice/stack_1:output:08conv1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv1/ActivityRegularizer/CastCast0conv1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
!conv1/ActivityRegularizer/truedivRealDiv2conv1/ActivityRegularizer/PartitionedCall:output:0"conv1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
maxpool1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_311121?
flatten/PartitionedCallPartitionedCall!maxpool1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_311176?
 mnist_tn/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0mnist_tn_311636mnist_tn_311638mnist_tn_311640*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_mnist_tn_layer_call_and_return_conditional_losses_311395?
,mnist_tn/ActivityRegularizer/PartitionedCallPartitionedCall)mnist_tn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU2*0J 8? *9
f4R2
0__inference_mnist_tn_activity_regularizer_311137{
"mnist_tn/ActivityRegularizer/ShapeShape)mnist_tn/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0mnist_tn/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2mnist_tn/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2mnist_tn/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*mnist_tn/ActivityRegularizer/strided_sliceStridedSlice+mnist_tn/ActivityRegularizer/Shape:output:09mnist_tn/ActivityRegularizer/strided_slice/stack:output:0;mnist_tn/ActivityRegularizer/strided_slice/stack_1:output:0;mnist_tn/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!mnist_tn/ActivityRegularizer/CastCast3mnist_tn/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$mnist_tn/ActivityRegularizer/truedivRealDiv5mnist_tn/ActivityRegularizer/PartitionedCall:output:0%mnist_tn/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
!out_layer/StatefulPartitionedCallStatefulPartitionedCall)mnist_tn/StatefulPartitionedCall:output:0out_layer_311651out_layer_311653*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_out_layer_layer_call_and_return_conditional_losses_311422y
IdentityIdentity*out_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
e

Identity_1Identity%conv1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_2Identity(mnist_tn/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^conv1/StatefulPartitionedCall!^mnist_tn/StatefulPartitionedCall"^out_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2D
 mnist_tn/StatefulPartitionedCall mnist_tn/StatefulPartitionedCall2F
!out_layer/StatefulPartitionedCall!out_layer/StatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_11
?	
?
+__inference_sequential_layer_call_fn_311618
input_11!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:?
	unknown_3:
	unknown_4:	?

	unknown_5:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
: : *)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_311578o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_11
?
?
H__inference_mnist_tn_layer_call_and_return_all_conditional_losses_312376

inputs
unknown:
	unknown_0:?
	unknown_1:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_mnist_tn_layer_call_and_return_conditional_losses_311395?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU2*0J 8? *9
f4R2
0__inference_mnist_tn_activity_regularizer_311137p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????X

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_out_layer_layer_call_fn_312385

inputs
unknown:	?

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_out_layer_layer_call_and_return_conditional_losses_311422o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_out_layer_layer_call_and_return_conditional_losses_312396

inputs1
matmul_readvariableop_resource:	?
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
A__inference_conv1_layer_call_and_return_conditional_losses_312407

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
+__inference_sequential_layer_call_fn_311760

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:?
	unknown_3:
	unknown_4:	?

	unknown_5:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
: : *)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_311578o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_311099
input_11I
/sequential_conv1_conv2d_readvariableop_resource:>
0sequential_conv1_biasadd_readvariableop_resource:K
5sequential_mnist_tn_loop_body_readvariableop_resource:M
7sequential_mnist_tn_loop_body_readvariableop_1_resource:?K
9sequential_mnist_tn_loop_body_add_readvariableop_resource:F
3sequential_out_layer_matmul_readvariableop_resource:	?
B
4sequential_out_layer_biasadd_readvariableop_resource:

identity??'sequential/conv1/BiasAdd/ReadVariableOp?&sequential/conv1/Conv2D/ReadVariableOp?,sequential/mnist_tn/loop_body/ReadVariableOp?.sequential/mnist_tn/loop_body/ReadVariableOp_1?0sequential/mnist_tn/loop_body/add/ReadVariableOp?+sequential/out_layer/BiasAdd/ReadVariableOp?*sequential/out_layer/MatMul/ReadVariableOp?
&sequential/conv1/Conv2D/ReadVariableOpReadVariableOp/sequential_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential/conv1/Conv2DConv2Dinput_11.sequential/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
'sequential/conv1/BiasAdd/ReadVariableOpReadVariableOp0sequential_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/conv1/BiasAddBiasAdd sequential/conv1/Conv2D:output:0/sequential/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z
sequential/conv1/ReluRelu!sequential/conv1/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
+sequential/conv1/ActivityRegularizer/SquareSquare#sequential/conv1/Relu:activations:0*
T0*/
_output_shapes
:??????????
*sequential/conv1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
(sequential/conv1/ActivityRegularizer/SumSum/sequential/conv1/ActivityRegularizer/Square:y:03sequential/conv1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: o
*sequential/conv1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
(sequential/conv1/ActivityRegularizer/mulMul3sequential/conv1/ActivityRegularizer/mul/x:output:01sequential/conv1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: }
*sequential/conv1/ActivityRegularizer/ShapeShape#sequential/conv1/Relu:activations:0*
T0*
_output_shapes
:?
8sequential/conv1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:sequential/conv1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:sequential/conv1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2sequential/conv1/ActivityRegularizer/strided_sliceStridedSlice3sequential/conv1/ActivityRegularizer/Shape:output:0Asequential/conv1/ActivityRegularizer/strided_slice/stack:output:0Csequential/conv1/ActivityRegularizer/strided_slice/stack_1:output:0Csequential/conv1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
)sequential/conv1/ActivityRegularizer/CastCast;sequential/conv1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
,sequential/conv1/ActivityRegularizer/truedivRealDiv,sequential/conv1/ActivityRegularizer/mul:z:0-sequential/conv1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
sequential/maxpool1/MaxPoolMaxPool#sequential/conv1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
sequential/flatten/ReshapeReshape$sequential/maxpool1/MaxPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????l
sequential/mnist_tn/ShapeShape#sequential/flatten/Reshape:output:0*
T0*
_output_shapes
:q
'sequential/mnist_tn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential/mnist_tn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential/mnist_tn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!sequential/mnist_tn/strided_sliceStridedSlice"sequential/mnist_tn/Shape:output:00sequential/mnist_tn/strided_slice/stack:output:02sequential/mnist_tn/strided_slice/stack_1:output:02sequential/mnist_tn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
sequential/mnist_tn/Rank/packedPack*sequential/mnist_tn/strided_slice:output:0*
N*
T0*
_output_shapes
:Z
sequential/mnist_tn/RankConst*
_output_shapes
: *
dtype0*
value	B :a
sequential/mnist_tn/range/startConst*
_output_shapes
: *
dtype0*
value	B : a
sequential/mnist_tn/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
sequential/mnist_tn/rangeRange(sequential/mnist_tn/range/start:output:0!sequential/mnist_tn/Rank:output:0(sequential/mnist_tn/range/delta:output:0*
_output_shapes
:
sequential/mnist_tn/Max/inputPack*sequential/mnist_tn/strided_slice:output:0*
N*
T0*
_output_shapes
:?
sequential/mnist_tn/MaxMax&sequential/mnist_tn/Max/input:output:0"sequential/mnist_tn/range:output:0*
T0*
_output_shapes
: |
:sequential/mnist_tn/loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : ?
4sequential/mnist_tn/loop_body/PlaceholderWithDefaultPlaceholderWithDefaultCsequential/mnist_tn/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: v
#sequential/mnist_tn/loop_body/ShapeShape#sequential/flatten/Reshape:output:0*
T0*
_output_shapes
:{
1sequential/mnist_tn/loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential/mnist_tn/loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential/mnist_tn/loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+sequential/mnist_tn/loop_body/strided_sliceStridedSlice,sequential/mnist_tn/loop_body/Shape:output:0:sequential/mnist_tn/loop_body/strided_slice/stack:output:0<sequential/mnist_tn/loop_body/strided_slice/stack_1:output:0<sequential/mnist_tn/loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'sequential/mnist_tn/loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :?
%sequential/mnist_tn/loop_body/GreaterGreater4sequential/mnist_tn/loop_body/strided_slice:output:00sequential/mnist_tn/loop_body/Greater/y:output:0*
T0*
_output_shapes
: j
(sequential/mnist_tn/loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : ?
&sequential/mnist_tn/loop_body/SelectV2SelectV2)sequential/mnist_tn/loop_body/Greater:z:0=sequential/mnist_tn/loop_body/PlaceholderWithDefault:output:01sequential/mnist_tn/loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: m
+sequential/mnist_tn/loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&sequential/mnist_tn/loop_body/GatherV2GatherV2#sequential/flatten/Reshape:output:0/sequential/mnist_tn/loop_body/SelectV2:output:04sequential/mnist_tn/loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes	
:?|
+sequential/mnist_tn/loop_body/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   ?
%sequential/mnist_tn/loop_body/ReshapeReshape/sequential/mnist_tn/loop_body/GatherV2:output:04sequential/mnist_tn/loop_body/Reshape/shape:output:0*
T0*
_output_shapes

:??
,sequential/mnist_tn/loop_body/ReadVariableOpReadVariableOp5sequential_mnist_tn_loop_body_readvariableop_resource*"
_output_shapes
:*
dtype0?
.sequential/mnist_tn/loop_body/ReadVariableOp_1ReadVariableOp7sequential_mnist_tn_loop_body_readvariableop_1_resource*"
_output_shapes
:?*
dtype0?
6sequential/mnist_tn/loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ?
1sequential/mnist_tn/loop_body/Tensordot/transpose	Transpose.sequential/mnist_tn/loop_body/Reshape:output:0?sequential/mnist_tn/loop_body/Tensordot/transpose/perm:output:0*
T0*
_output_shapes

:??
8sequential/mnist_tn/loop_body/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
3sequential/mnist_tn/loop_body/Tensordot/transpose_1	Transpose4sequential/mnist_tn/loop_body/ReadVariableOp:value:0Asequential/mnist_tn/loop_body/Tensordot/transpose_1/perm:output:0*
T0*"
_output_shapes
:?
5sequential/mnist_tn/loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   ?
/sequential/mnist_tn/loop_body/Tensordot/ReshapeReshape7sequential/mnist_tn/loop_body/Tensordot/transpose_1:y:0>sequential/mnist_tn/loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	??
.sequential/mnist_tn/loop_body/Tensordot/MatMulMatMul5sequential/mnist_tn/loop_body/Tensordot/transpose:y:08sequential/mnist_tn/loop_body/Tensordot/Reshape:output:0*
T0*
_output_shapes
:	???
-sequential/mnist_tn/loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"?         ?
'sequential/mnist_tn/loop_body/TensordotReshape8sequential/mnist_tn/loop_body/Tensordot/MatMul:product:06sequential/mnist_tn/loop_body/Tensordot/shape:output:0*
T0*"
_output_shapes
:??
7sequential/mnist_tn/loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?  ?
1sequential/mnist_tn/loop_body/Tensordot_1/ReshapeReshape6sequential/mnist_tn/loop_body/ReadVariableOp_1:value:0@sequential/mnist_tn/loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes
:	??
8sequential/mnist_tn/loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
3sequential/mnist_tn/loop_body/Tensordot_1/transpose	Transpose0sequential/mnist_tn/loop_body/Tensordot:output:0Asequential/mnist_tn/loop_body/Tensordot_1/transpose/perm:output:0*
T0*"
_output_shapes
:??
9sequential/mnist_tn/loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"?     ?
3sequential/mnist_tn/loop_body/Tensordot_1/Reshape_1Reshape7sequential/mnist_tn/loop_body/Tensordot_1/transpose:y:0Bsequential/mnist_tn/loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	??
0sequential/mnist_tn/loop_body/Tensordot_1/MatMulMatMul:sequential/mnist_tn/loop_body/Tensordot_1/Reshape:output:0<sequential/mnist_tn/loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes

:}
,sequential/mnist_tn/loop_body/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ?
'sequential/mnist_tn/loop_body/transpose	Transpose:sequential/mnist_tn/loop_body/Tensordot_1/MatMul:product:05sequential/mnist_tn/loop_body/transpose/perm:output:0*
T0*
_output_shapes

:?
0sequential/mnist_tn/loop_body/add/ReadVariableOpReadVariableOp9sequential_mnist_tn_loop_body_add_readvariableop_resource*
_output_shapes

:*
dtype0?
!sequential/mnist_tn/loop_body/addAddV2+sequential/mnist_tn/loop_body/transpose:y:08sequential/mnist_tn/loop_body/add/ReadVariableOp:value:0*
T0*
_output_shapes

:p
&sequential/mnist_tn/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
 sequential/mnist_tn/pfor/ReshapeReshape sequential/mnist_tn/Max:output:0/sequential/mnist_tn/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:f
$sequential/mnist_tn/pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : f
$sequential/mnist_tn/pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
sequential/mnist_tn/pfor/rangeRange-sequential/mnist_tn/pfor/range/start:output:0 sequential/mnist_tn/Max:output:0-sequential/mnist_tn/pfor/range/delta:output:0*#
_output_shapes
:?????????r
0sequential/mnist_tn/loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : s
1sequential/mnist_tn/loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
/sequential/mnist_tn/loop_body/SelectV2/pfor/addAddV29sequential/mnist_tn/loop_body/SelectV2/pfor/Rank:output:0:sequential/mnist_tn/loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: t
2sequential/mnist_tn/loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :t
2sequential/mnist_tn/loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : u
3sequential/mnist_tn/loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
1sequential/mnist_tn/loop_body/SelectV2/pfor/add_1AddV2;sequential/mnist_tn/loop_body/SelectV2/pfor/Rank_2:output:0<sequential/mnist_tn/loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: ?
3sequential/mnist_tn/loop_body/SelectV2/pfor/MaximumMaximum;sequential/mnist_tn/loop_body/SelectV2/pfor/Rank_1:output:03sequential/mnist_tn/loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: ?
5sequential/mnist_tn/loop_body/SelectV2/pfor/Maximum_1Maximum5sequential/mnist_tn/loop_body/SelectV2/pfor/add_1:z:07sequential/mnist_tn/loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: ?
1sequential/mnist_tn/loop_body/SelectV2/pfor/ShapeShape'sequential/mnist_tn/pfor/range:output:0*
T0*
_output_shapes
:?
/sequential/mnist_tn/loop_body/SelectV2/pfor/subSub9sequential/mnist_tn/loop_body/SelectV2/pfor/Maximum_1:z:0;sequential/mnist_tn/loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: ?
9sequential/mnist_tn/loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
3sequential/mnist_tn/loop_body/SelectV2/pfor/ReshapeReshape3sequential/mnist_tn/loop_body/SelectV2/pfor/sub:z:0Bsequential/mnist_tn/loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:?
6sequential/mnist_tn/loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:?
0sequential/mnist_tn/loop_body/SelectV2/pfor/TileTile?sequential/mnist_tn/loop_body/SelectV2/pfor/Tile/input:output:0<sequential/mnist_tn/loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: ?
?sequential/mnist_tn/loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Asequential/mnist_tn/loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Asequential/mnist_tn/loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9sequential/mnist_tn/loop_body/SelectV2/pfor/strided_sliceStridedSlice:sequential/mnist_tn/loop_body/SelectV2/pfor/Shape:output:0Hsequential/mnist_tn/loop_body/SelectV2/pfor/strided_slice/stack:output:0Jsequential/mnist_tn/loop_body/SelectV2/pfor/strided_slice/stack_1:output:0Jsequential/mnist_tn/loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Asequential/mnist_tn/loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Csequential/mnist_tn/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Csequential/mnist_tn/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;sequential/mnist_tn/loop_body/SelectV2/pfor/strided_slice_1StridedSlice:sequential/mnist_tn/loop_body/SelectV2/pfor/Shape:output:0Jsequential/mnist_tn/loop_body/SelectV2/pfor/strided_slice_1/stack:output:0Lsequential/mnist_tn/loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:0Lsequential/mnist_tn/loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_masky
7sequential/mnist_tn/loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2sequential/mnist_tn/loop_body/SelectV2/pfor/concatConcatV2Bsequential/mnist_tn/loop_body/SelectV2/pfor/strided_slice:output:09sequential/mnist_tn/loop_body/SelectV2/pfor/Tile:output:0Dsequential/mnist_tn/loop_body/SelectV2/pfor/strided_slice_1:output:0@sequential/mnist_tn/loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
5sequential/mnist_tn/loop_body/SelectV2/pfor/Reshape_1Reshape'sequential/mnist_tn/pfor/range:output:0;sequential/mnist_tn/loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:??????????
4sequential/mnist_tn/loop_body/SelectV2/pfor/SelectV2SelectV2)sequential/mnist_tn/loop_body/Greater:z:0>sequential/mnist_tn/loop_body/SelectV2/pfor/Reshape_1:output:01sequential/mnist_tn/loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:?????????{
9sequential/mnist_tn/loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4sequential/mnist_tn/loop_body/GatherV2/pfor/GatherV2GatherV2#sequential/flatten/Reshape:output:0=sequential/mnist_tn/loop_body/SelectV2/pfor/SelectV2:output:0Bsequential/mnist_tn/loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*(
_output_shapes
:??????????x
6sequential/mnist_tn/loop_body/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1sequential/mnist_tn/loop_body/Reshape/pfor/concatConcatV2)sequential/mnist_tn/pfor/Reshape:output:04sequential/mnist_tn/loop_body/Reshape/shape:output:0?sequential/mnist_tn/loop_body/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
2sequential/mnist_tn/loop_body/Reshape/pfor/ReshapeReshape=sequential/mnist_tn/loop_body/GatherV2/pfor/GatherV2:output:0:sequential/mnist_tn/loop_body/Reshape/pfor/concat:output:0*
T0*+
_output_shapes
:??????????~
<sequential/mnist_tn/loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
:sequential/mnist_tn/loop_body/Tensordot/transpose/pfor/addAddV2?sequential/mnist_tn/loop_body/Tensordot/transpose/perm:output:0Esequential/mnist_tn/loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:?
Fsequential/mnist_tn/loop_body/Tensordot/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: ?
Bsequential/mnist_tn/loop_body/Tensordot/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
=sequential/mnist_tn/loop_body/Tensordot/transpose/pfor/concatConcatV2Osequential/mnist_tn/loop_body/Tensordot/transpose/pfor/concat/values_0:output:0>sequential/mnist_tn/loop_body/Tensordot/transpose/pfor/add:z:0Ksequential/mnist_tn/loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
@sequential/mnist_tn/loop_body/Tensordot/transpose/pfor/Transpose	Transpose;sequential/mnist_tn/loop_body/Reshape/pfor/Reshape:output:0Fsequential/mnist_tn/loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*+
_output_shapes
:???????????
9sequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/ShapeShapeDsequential/mnist_tn/loop_body/Tensordot/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:?
Csequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
9sequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/splitSplitLsequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/split/split_dim:output:0Bsequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Shape:output:0*
T0*&
_output_shapes
:::*
	num_split?
Asequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Csequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
;sequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/ReshapeReshapeBsequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/split:output:0Lsequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1:output:0*
T0*
_output_shapes
: ?
Csequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Esequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
=sequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_1ReshapeBsequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/split:output:1Nsequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ?
Csequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Esequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
=sequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_2ReshapeBsequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/split:output:2Nsequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ?
7sequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/mulMulDsequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape:output:0Fsequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: ?
Csequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack;sequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/mul:z:0Fsequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:?
=sequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_3ReshapeDsequential/mnist_tn/loop_body/Tensordot/transpose/pfor/Transpose:y:0Lsequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:???????????????????
:sequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/MatMulMatMulFsequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_3:output:08sequential/mnist_tn/loop_body/Tensordot/Reshape:output:0*
T0*(
_output_shapes
:???????????
Esequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
??????????
Csequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePackDsequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape:output:0Fsequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0Nsequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:?
=sequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_4ReshapeDsequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/MatMul:product:0Lsequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*,
_output_shapes
:???????????z
8sequential/mnist_tn/loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
3sequential/mnist_tn/loop_body/Tensordot/pfor/concatConcatV2)sequential/mnist_tn/pfor/Reshape:output:06sequential/mnist_tn/loop_body/Tensordot/shape:output:0Asequential/mnist_tn/loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
4sequential/mnist_tn/loop_body/Tensordot/pfor/ReshapeReshapeFsequential/mnist_tn/loop_body/Tensordot/MatMul/pfor/Reshape_4:output:0<sequential/mnist_tn/loop_body/Tensordot/pfor/concat:output:0*
T0*/
_output_shapes
:???????????
>sequential/mnist_tn/loop_body/Tensordot_1/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
<sequential/mnist_tn/loop_body/Tensordot_1/transpose/pfor/addAddV2Asequential/mnist_tn/loop_body/Tensordot_1/transpose/perm:output:0Gsequential/mnist_tn/loop_body/Tensordot_1/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:?
Hsequential/mnist_tn/loop_body/Tensordot_1/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: ?
Dsequential/mnist_tn/loop_body/Tensordot_1/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
?sequential/mnist_tn/loop_body/Tensordot_1/transpose/pfor/concatConcatV2Qsequential/mnist_tn/loop_body/Tensordot_1/transpose/pfor/concat/values_0:output:0@sequential/mnist_tn/loop_body/Tensordot_1/transpose/pfor/add:z:0Msequential/mnist_tn/loop_body/Tensordot_1/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Bsequential/mnist_tn/loop_body/Tensordot_1/transpose/pfor/Transpose	Transpose=sequential/mnist_tn/loop_body/Tensordot/pfor/Reshape:output:0Hsequential/mnist_tn/loop_body/Tensordot_1/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:???????????
Dsequential/mnist_tn/loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
?sequential/mnist_tn/loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2)sequential/mnist_tn/pfor/Reshape:output:0Bsequential/mnist_tn/loop_body/Tensordot_1/Reshape_1/shape:output:0Msequential/mnist_tn/loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
@sequential/mnist_tn/loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshapeFsequential/mnist_tn/loop_body/Tensordot_1/transpose/pfor/Transpose:y:0Hsequential/mnist_tn/loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:???????????
;sequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/ShapeShapeIsequential/mnist_tn/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:?
Isequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Ksequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Ksequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Csequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSliceDsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Rsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0Tsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0Tsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Ksequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Msequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Msequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Esequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSliceDsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Tsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Vsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Vsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
=sequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/MinimumMinimumLsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0Nsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: 
=sequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :?
;sequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/EqualEqualAsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Minimum:z:0Fsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: ?
>sequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          ?
>sequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          ?
<sequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/SelectSelect?sequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Equal:z:0Gsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Select/t:output:0Gsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:?
Ksequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Msequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Msequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Esequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSliceDsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Tsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Vsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Vsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Ksequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Msequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Msequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Esequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSliceDsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Tsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Vsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Vsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Ksequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Msequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Msequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Esequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSliceDsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Tsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Vsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Vsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
;sequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/stackPackNsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0Nsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0Nsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
?sequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/transpose	TransposeIsequential/mnist_tn/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0Esequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'????????????????????????????
=sequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/ReshapeReshapeCsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/transpose:y:0Dsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:???????????
=sequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Shape_1ShapeFsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:?
Esequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
;sequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/splitSplitNsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/split/split_dim:output:0Fsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split?
Esequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Gsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
?sequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_1ReshapeDsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/split:output:0Psequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ?
Esequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Gsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
?sequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_2ReshapeDsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/split:output:1Psequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ?
Esequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Gsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
?sequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_3ReshapeDsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/split:output:2Psequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ?
9sequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/mulMulHsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0Hsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ?
Esequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePackHsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:0=sequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:?
?sequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_4ReshapeFsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0Nsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:???????????????????
<sequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul:sequential/mnist_tn/loop_body/Tensordot_1/Reshape:output:0Hsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:??????????
Gsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
??????????
Esequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePackPsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:0Hsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0Hsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:?
?sequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_5ReshapeFsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0Nsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'????????????????????????????
Fsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
Asequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/transpose_1	TransposeHsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0Osequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????t
2sequential/mnist_tn/loop_body/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
0sequential/mnist_tn/loop_body/transpose/pfor/addAddV25sequential/mnist_tn/loop_body/transpose/perm:output:0;sequential/mnist_tn/loop_body/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:?
<sequential/mnist_tn/loop_body/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: z
8sequential/mnist_tn/loop_body/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
3sequential/mnist_tn/loop_body/transpose/pfor/concatConcatV2Esequential/mnist_tn/loop_body/transpose/pfor/concat/values_0:output:04sequential/mnist_tn/loop_body/transpose/pfor/add:z:0Asequential/mnist_tn/loop_body/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
6sequential/mnist_tn/loop_body/transpose/pfor/Transpose	TransposeEsequential/mnist_tn/loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:0<sequential/mnist_tn/loop_body/transpose/pfor/concat:output:0*
T0*+
_output_shapes
:?????????m
+sequential/mnist_tn/loop_body/add/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :o
-sequential/mnist_tn/loop_body/add/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :n
,sequential/mnist_tn/loop_body/add/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
*sequential/mnist_tn/loop_body/add/pfor/addAddV26sequential/mnist_tn/loop_body/add/pfor/Rank_1:output:05sequential/mnist_tn/loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: ?
.sequential/mnist_tn/loop_body/add/pfor/MaximumMaximum.sequential/mnist_tn/loop_body/add/pfor/add:z:04sequential/mnist_tn/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: ?
,sequential/mnist_tn/loop_body/add/pfor/ShapeShape:sequential/mnist_tn/loop_body/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:?
*sequential/mnist_tn/loop_body/add/pfor/subSub2sequential/mnist_tn/loop_body/add/pfor/Maximum:z:04sequential/mnist_tn/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: ~
4sequential/mnist_tn/loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
.sequential/mnist_tn/loop_body/add/pfor/ReshapeReshape.sequential/mnist_tn/loop_body/add/pfor/sub:z:0=sequential/mnist_tn/loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:{
1sequential/mnist_tn/loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:?
+sequential/mnist_tn/loop_body/add/pfor/TileTile:sequential/mnist_tn/loop_body/add/pfor/Tile/input:output:07sequential/mnist_tn/loop_body/add/pfor/Reshape:output:0*
T0*
_output_shapes
: ?
:sequential/mnist_tn/loop_body/add/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<sequential/mnist_tn/loop_body/add/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<sequential/mnist_tn/loop_body/add/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4sequential/mnist_tn/loop_body/add/pfor/strided_sliceStridedSlice5sequential/mnist_tn/loop_body/add/pfor/Shape:output:0Csequential/mnist_tn/loop_body/add/pfor/strided_slice/stack:output:0Esequential/mnist_tn/loop_body/add/pfor/strided_slice/stack_1:output:0Esequential/mnist_tn/loop_body/add/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
<sequential/mnist_tn/loop_body/add/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
>sequential/mnist_tn/loop_body/add/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
>sequential/mnist_tn/loop_body/add/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6sequential/mnist_tn/loop_body/add/pfor/strided_slice_1StridedSlice5sequential/mnist_tn/loop_body/add/pfor/Shape:output:0Esequential/mnist_tn/loop_body/add/pfor/strided_slice_1/stack:output:0Gsequential/mnist_tn/loop_body/add/pfor/strided_slice_1/stack_1:output:0Gsequential/mnist_tn/loop_body/add/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskt
2sequential/mnist_tn/loop_body/add/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-sequential/mnist_tn/loop_body/add/pfor/concatConcatV2=sequential/mnist_tn/loop_body/add/pfor/strided_slice:output:04sequential/mnist_tn/loop_body/add/pfor/Tile:output:0?sequential/mnist_tn/loop_body/add/pfor/strided_slice_1:output:0;sequential/mnist_tn/loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
0sequential/mnist_tn/loop_body/add/pfor/Reshape_1Reshape:sequential/mnist_tn/loop_body/transpose/pfor/Transpose:y:06sequential/mnist_tn/loop_body/add/pfor/concat:output:0*
T0*+
_output_shapes
:??????????
,sequential/mnist_tn/loop_body/add/pfor/AddV2AddV29sequential/mnist_tn/loop_body/add/pfor/Reshape_1:output:08sequential/mnist_tn/loop_body/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????r
!sequential/mnist_tn/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
sequential/mnist_tn/ReshapeReshape0sequential/mnist_tn/loop_body/add/pfor/AddV2:z:0*sequential/mnist_tn/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????y
sequential/mnist_tn/ReluRelu$sequential/mnist_tn/Reshape:output:0*
T0*(
_output_shapes
:???????????
.sequential/mnist_tn/ActivityRegularizer/SquareSquare&sequential/mnist_tn/Relu:activations:0*
T0*(
_output_shapes
:??????????~
-sequential/mnist_tn/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
+sequential/mnist_tn/ActivityRegularizer/SumSum2sequential/mnist_tn/ActivityRegularizer/Square:y:06sequential/mnist_tn/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: r
-sequential/mnist_tn/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+sequential/mnist_tn/ActivityRegularizer/mulMul6sequential/mnist_tn/ActivityRegularizer/mul/x:output:04sequential/mnist_tn/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: ?
-sequential/mnist_tn/ActivityRegularizer/ShapeShape&sequential/mnist_tn/Relu:activations:0*
T0*
_output_shapes
:?
;sequential/mnist_tn/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
=sequential/mnist_tn/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=sequential/mnist_tn/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5sequential/mnist_tn/ActivityRegularizer/strided_sliceStridedSlice6sequential/mnist_tn/ActivityRegularizer/Shape:output:0Dsequential/mnist_tn/ActivityRegularizer/strided_slice/stack:output:0Fsequential/mnist_tn/ActivityRegularizer/strided_slice/stack_1:output:0Fsequential/mnist_tn/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
,sequential/mnist_tn/ActivityRegularizer/CastCast>sequential/mnist_tn/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
/sequential/mnist_tn/ActivityRegularizer/truedivRealDiv/sequential/mnist_tn/ActivityRegularizer/mul:z:00sequential/mnist_tn/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
*sequential/out_layer/MatMul/ReadVariableOpReadVariableOp3sequential_out_layer_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
sequential/out_layer/MatMulMatMul&sequential/mnist_tn/Relu:activations:02sequential/out_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
+sequential/out_layer/BiasAdd/ReadVariableOpReadVariableOp4sequential_out_layer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
sequential/out_layer/BiasAddBiasAdd%sequential/out_layer/MatMul:product:03sequential/out_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
sequential/out_layer/SoftmaxSoftmax%sequential/out_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
u
IdentityIdentity&sequential/out_layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp(^sequential/conv1/BiasAdd/ReadVariableOp'^sequential/conv1/Conv2D/ReadVariableOp-^sequential/mnist_tn/loop_body/ReadVariableOp/^sequential/mnist_tn/loop_body/ReadVariableOp_11^sequential/mnist_tn/loop_body/add/ReadVariableOp,^sequential/out_layer/BiasAdd/ReadVariableOp+^sequential/out_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : 2R
'sequential/conv1/BiasAdd/ReadVariableOp'sequential/conv1/BiasAdd/ReadVariableOp2P
&sequential/conv1/Conv2D/ReadVariableOp&sequential/conv1/Conv2D/ReadVariableOp2\
,sequential/mnist_tn/loop_body/ReadVariableOp,sequential/mnist_tn/loop_body/ReadVariableOp2`
.sequential/mnist_tn/loop_body/ReadVariableOp_1.sequential/mnist_tn/loop_body/ReadVariableOp_12d
0sequential/mnist_tn/loop_body/add/ReadVariableOp0sequential/mnist_tn/loop_body/add/ReadVariableOp2Z
+sequential/out_layer/BiasAdd/ReadVariableOp+sequential/out_layer/BiasAdd/ReadVariableOp2X
*sequential/out_layer/MatMul/ReadVariableOp*sequential/out_layer/MatMul/ReadVariableOp:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_11
?

?
E__inference_out_layer_layer_call_and_return_conditional_losses_311422

inputs1
matmul_readvariableop_resource:	?
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_conv1_layer_call_fn_312314

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_311155w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
A__inference_conv1_layer_call_and_return_conditional_losses_311155

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
0__inference_mnist_tn_activity_regularizer_311137
x
identity6
SquareSquarex*
T0*
_output_shapes
:9
RankRank
Square:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????G
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
?
`
D__inference_maxpool1_layer_call_and_return_conditional_losses_311121

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_311176

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?1
?
F__inference_sequential_layer_call_and_return_conditional_losses_311700
input_11&
conv1_311662:
conv1_311664:%
mnist_tn_311677:%
mnist_tn_311679:?!
mnist_tn_311681:#
out_layer_311692:	?

out_layer_311694:

identity

identity_1

identity_2??conv1/StatefulPartitionedCall? mnist_tn/StatefulPartitionedCall?!out_layer/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallinput_11conv1_311662conv1_311664*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_311155?
)conv1/ActivityRegularizer/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU2*0J 8? *6
f1R/
-__inference_conv1_activity_regularizer_311112u
conv1/ActivityRegularizer/ShapeShape&conv1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-conv1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/conv1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/conv1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'conv1/ActivityRegularizer/strided_sliceStridedSlice(conv1/ActivityRegularizer/Shape:output:06conv1/ActivityRegularizer/strided_slice/stack:output:08conv1/ActivityRegularizer/strided_slice/stack_1:output:08conv1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv1/ActivityRegularizer/CastCast0conv1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
!conv1/ActivityRegularizer/truedivRealDiv2conv1/ActivityRegularizer/PartitionedCall:output:0"conv1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
maxpool1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_311121?
flatten/PartitionedCallPartitionedCall!maxpool1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_311176?
 mnist_tn/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0mnist_tn_311677mnist_tn_311679mnist_tn_311681*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_mnist_tn_layer_call_and_return_conditional_losses_311395?
,mnist_tn/ActivityRegularizer/PartitionedCallPartitionedCall)mnist_tn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU2*0J 8? *9
f4R2
0__inference_mnist_tn_activity_regularizer_311137{
"mnist_tn/ActivityRegularizer/ShapeShape)mnist_tn/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0mnist_tn/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2mnist_tn/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2mnist_tn/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*mnist_tn/ActivityRegularizer/strided_sliceStridedSlice+mnist_tn/ActivityRegularizer/Shape:output:09mnist_tn/ActivityRegularizer/strided_slice/stack:output:0;mnist_tn/ActivityRegularizer/strided_slice/stack_1:output:0;mnist_tn/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!mnist_tn/ActivityRegularizer/CastCast3mnist_tn/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$mnist_tn/ActivityRegularizer/truedivRealDiv5mnist_tn/ActivityRegularizer/PartitionedCall:output:0%mnist_tn/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
!out_layer/StatefulPartitionedCallStatefulPartitionedCall)mnist_tn/StatefulPartitionedCall:output:0out_layer_311692out_layer_311694*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_out_layer_layer_call_and_return_conditional_losses_311422y
IdentityIdentity*out_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
e

Identity_1Identity%conv1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_2Identity(mnist_tn/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^conv1/StatefulPartitionedCall!^mnist_tn/StatefulPartitionedCall"^out_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2D
 mnist_tn/StatefulPartitionedCall mnist_tn/StatefulPartitionedCall2F
!out_layer/StatefulPartitionedCall!out_layer/StatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
input_11
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_312346

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_119
serving_default_input_11:0?????????=
	out_layer0
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:?v
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	#a_var
	$b_var
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
?

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
?
4iter

5beta_1

6beta_2
	7decay
8learning_ratemgmh#mi$mj%mk,ml-mmvnvo#vp$vq%vr,vs-vt"
	optimizer
Q
0
1
#2
$3
%4
,5
-6"
trackable_list_wrapper
Q
0
1
#2
$3
%4
,5
-6"
trackable_list_wrapper
 "
trackable_list_wrapper
?
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_sequential_layer_call_fn_311450
+__inference_sequential_layer_call_fn_311739
+__inference_sequential_layer_call_fn_311760
+__inference_sequential_layer_call_fn_311618?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_sequential_layer_call_and_return_conditional_losses_312022
F__inference_sequential_layer_call_and_return_conditional_losses_312284
F__inference_sequential_layer_call_and_return_conditional_losses_311659
F__inference_sequential_layer_call_and_return_conditional_losses_311700?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_311099input_11"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
>serving_default"
signature_map
&:$2conv1/kernel
:2
conv1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
Dactivity_regularizer_fn
*&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_conv1_layer_call_fn_312314?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv1_layer_call_and_return_all_conditional_losses_312325?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_maxpool1_layer_call_fn_312330?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_maxpool1_layer_call_and_return_conditional_losses_312335?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_flatten_layer_call_fn_312340?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_flatten_layer_call_and_return_conditional_losses_312346?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:2a
:?2b
:2bias
5
#0
$1
%2"
trackable_list_wrapper
5
#0
$1
%2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
Uactivity_regularizer_fn
*+&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_mnist_tn_layer_call_fn_312363?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_mnist_tn_layer_call_and_return_all_conditional_losses_312376?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
#:!	?
2out_layer/kernel
:
2out_layer/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_out_layer_layer_call_fn_312385?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_out_layer_layer_call_and_return_conditional_losses_312396?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
$__inference_signature_wrapper_312305input_11"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?2?
-__inference_conv1_activity_regularizer_311112?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
A__inference_conv1_layer_call_and_return_conditional_losses_312407?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?2?
0__inference_mnist_tn_activity_regularizer_311137?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
D__inference_mnist_tn_layer_call_and_return_conditional_losses_312624?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
	^total
	_count
`	variables
a	keras_api"
_tf_keras_metric
q
b
thresholds
ctrue_positives
dfalse_positives
e	variables
f	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
^0
_1"
trackable_list_wrapper
-
`	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
.
c0
d1"
trackable_list_wrapper
-
e	variables"
_generic_user_object
+:)2Adam/conv1/kernel/m
:2Adam/conv1/bias/m
:2Adam/a/m
:?2Adam/b/m
:2Adam/bias/m
(:&	?
2Adam/out_layer/kernel/m
!:
2Adam/out_layer/bias/m
+:)2Adam/conv1/kernel/v
:2Adam/conv1/bias/v
:2Adam/a/v
:?2Adam/b/v
:2Adam/bias/v
(:&	?
2Adam/out_layer/kernel/v
!:
2Adam/out_layer/bias/v?
!__inference__wrapped_model_311099{#$%,-9?6
/?,
*?'
input_11?????????
? "5?2
0
	out_layer#? 
	out_layer?????????
W
-__inference_conv1_activity_regularizer_311112&?
?
?	
x
? "? ?
E__inference_conv1_layer_call_and_return_all_conditional_losses_312325z7?4
-?*
(?%
inputs?????????
? ";?8
#? 
0?????????
?
?	
1/0 ?
A__inference_conv1_layer_call_and_return_conditional_losses_312407l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
&__inference_conv1_layer_call_fn_312314_7?4
-?*
(?%
inputs?????????
? " ???????????
C__inference_flatten_layer_call_and_return_conditional_losses_312346a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
(__inference_flatten_layer_call_fn_312340T7?4
-?*
(?%
inputs?????????
? "????????????
D__inference_maxpool1_layer_call_and_return_conditional_losses_312335?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
)__inference_maxpool1_layer_call_fn_312330?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????Z
0__inference_mnist_tn_activity_regularizer_311137&?
?
?	
x
? "? ?
H__inference_mnist_tn_layer_call_and_return_all_conditional_losses_312376m#$%0?-
&?#
!?
inputs??????????
? "4?1
?
0??????????
?
?	
1/0 ?
D__inference_mnist_tn_layer_call_and_return_conditional_losses_312624_#$%0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
)__inference_mnist_tn_layer_call_fn_312363R#$%0?-
&?#
!?
inputs??????????
? "????????????
E__inference_out_layer_layer_call_and_return_conditional_losses_312396],-0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? ~
*__inference_out_layer_layer_call_fn_312385P,-0?-
&?#
!?
inputs??????????
? "??????????
?
F__inference_sequential_layer_call_and_return_conditional_losses_311659?#$%,-A?>
7?4
*?'
input_11?????????
p 

 
? "A?>
?
0?????????

?
?	
1/0 
?	
1/1 ?
F__inference_sequential_layer_call_and_return_conditional_losses_311700?#$%,-A?>
7?4
*?'
input_11?????????
p

 
? "A?>
?
0?????????

?
?	
1/0 
?	
1/1 ?
F__inference_sequential_layer_call_and_return_conditional_losses_312022?#$%,-??<
5?2
(?%
inputs?????????
p 

 
? "A?>
?
0?????????

?
?	
1/0 
?	
1/1 ?
F__inference_sequential_layer_call_and_return_conditional_losses_312284?#$%,-??<
5?2
(?%
inputs?????????
p

 
? "A?>
?
0?????????

?
?	
1/0 
?	
1/1 ?
+__inference_sequential_layer_call_fn_311450f#$%,-A?>
7?4
*?'
input_11?????????
p 

 
? "??????????
?
+__inference_sequential_layer_call_fn_311618f#$%,-A?>
7?4
*?'
input_11?????????
p

 
? "??????????
?
+__inference_sequential_layer_call_fn_311739d#$%,-??<
5?2
(?%
inputs?????????
p 

 
? "??????????
?
+__inference_sequential_layer_call_fn_311760d#$%,-??<
5?2
(?%
inputs?????????
p

 
? "??????????
?
$__inference_signature_wrapper_312305?#$%,-E?B
? 
;?8
6
input_11*?'
input_11?????????"5?2
0
	out_layer#? 
	out_layer?????????
