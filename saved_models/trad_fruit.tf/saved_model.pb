
òÃ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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
8
Const
output"dtype"
valuetensor"
dtypetype

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

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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
7
Square
x"T
y"T"
Ttype:
2	
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

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68¬£

conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:*
dtype0
|
conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1/kernel
u
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*&
_output_shapes
:*
dtype0
l

conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
conv1/bias
e
conv1/bias/Read/ReadVariableOpReadVariableOp
conv1/bias*
_output_shapes
:*
dtype0
}
conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2/kernel
v
 conv2/kernel/Read/ReadVariableOpReadVariableOpconv2/kernel*'
_output_shapes
:*
dtype0
m

conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
conv2/bias
f
conv2/bias/Read/ReadVariableOpReadVariableOp
conv2/bias*
_output_shapes	
:*
dtype0
x
dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ò@*
shared_namedense1/kernel
q
!dense1/kernel/Read/ReadVariableOpReadVariableOpdense1/kernel* 
_output_shapes
:
ò@*
dtype0
n
dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense1/bias
g
dense1/bias/Read/ReadVariableOpReadVariableOpdense1/bias*
_output_shapes
:@*
dtype0
}
out_dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*!
shared_nameout_dense/kernel
v
$out_dense/kernel/Read/ReadVariableOpReadVariableOpout_dense/kernel*
_output_shapes
:	@*
dtype0
u
out_dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameout_dense/bias
n
"out_dense/bias/Read/ReadVariableOpReadVariableOpout_dense/bias*
_output_shapes	
:*
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_3/kernel/m

*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/m
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes
:*
dtype0

Adam/conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/conv1/kernel/m

'Adam/conv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1/kernel/m*&
_output_shapes
:*
dtype0
z
Adam/conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/conv1/bias/m
s
%Adam/conv1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1/bias/m*
_output_shapes
:*
dtype0

Adam/conv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/conv2/kernel/m

'Adam/conv2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2/kernel/m*'
_output_shapes
:*
dtype0
{
Adam/conv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/conv2/bias/m
t
%Adam/conv2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2/bias/m*
_output_shapes	
:*
dtype0

Adam/dense1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ò@*%
shared_nameAdam/dense1/kernel/m

(Adam/dense1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense1/kernel/m* 
_output_shapes
:
ò@*
dtype0
|
Adam/dense1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/dense1/bias/m
u
&Adam/dense1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense1/bias/m*
_output_shapes
:@*
dtype0

Adam/out_dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*(
shared_nameAdam/out_dense/kernel/m

+Adam/out_dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/out_dense/kernel/m*
_output_shapes
:	@*
dtype0

Adam/out_dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/out_dense/bias/m
|
)Adam/out_dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/out_dense/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_3/kernel/v

*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/v
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes
:*
dtype0

Adam/conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/conv1/kernel/v

'Adam/conv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1/kernel/v*&
_output_shapes
:*
dtype0
z
Adam/conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/conv1/bias/v
s
%Adam/conv1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1/bias/v*
_output_shapes
:*
dtype0

Adam/conv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/conv2/kernel/v

'Adam/conv2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2/kernel/v*'
_output_shapes
:*
dtype0
{
Adam/conv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/conv2/bias/v
t
%Adam/conv2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2/bias/v*
_output_shapes	
:*
dtype0

Adam/dense1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ò@*%
shared_nameAdam/dense1/kernel/v

(Adam/dense1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense1/kernel/v* 
_output_shapes
:
ò@*
dtype0
|
Adam/dense1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/dense1/bias/v
u
&Adam/dense1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense1/bias/v*
_output_shapes
:@*
dtype0

Adam/out_dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*(
shared_nameAdam/out_dense/kernel/v

+Adam/out_dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/out_dense/kernel/v*
_output_shapes
:	@*
dtype0

Adam/out_dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/out_dense/bias/v
|
)Adam/out_dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/out_dense/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_3/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/conv2d_3/kernel/vhat

-Adam/conv2d_3/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/vhat*&
_output_shapes
:*
dtype0

Adam/conv2d_3/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_3/bias/vhat

+Adam/conv2d_3/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/vhat*
_output_shapes
:*
dtype0

Adam/conv1/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv1/kernel/vhat

*Adam/conv1/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/conv1/kernel/vhat*&
_output_shapes
:*
dtype0

Adam/conv1/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1/bias/vhat
y
(Adam/conv1/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/conv1/bias/vhat*
_output_shapes
:*
dtype0

Adam/conv2/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2/kernel/vhat

*Adam/conv2/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/conv2/kernel/vhat*'
_output_shapes
:*
dtype0

Adam/conv2/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2/bias/vhat
z
(Adam/conv2/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/conv2/bias/vhat*
_output_shapes	
:*
dtype0

Adam/dense1/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ò@*(
shared_nameAdam/dense1/kernel/vhat

+Adam/dense1/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/dense1/kernel/vhat* 
_output_shapes
:
ò@*
dtype0

Adam/dense1/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense1/bias/vhat
{
)Adam/dense1/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/dense1/bias/vhat*
_output_shapes
:@*
dtype0

Adam/out_dense/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*+
shared_nameAdam/out_dense/kernel/vhat

.Adam/out_dense/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/out_dense/kernel/vhat*
_output_shapes
:	@*
dtype0

Adam/out_dense/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/out_dense/bias/vhat

,Adam/out_dense/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/out_dense/bias/vhat*
_output_shapes	
:*
dtype0

NoOpNoOp
a
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ë`
valueÁ`B¾` B·`
Ð
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses*

$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses* 
¦

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses*

2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses* 

8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses* 
¥
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B_random_generator
C__call__
*D&call_and_return_all_conditional_losses* 
¦

Ekernel
Fbias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses*
¥
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q_random_generator
R__call__
*S&call_and_return_all_conditional_losses* 
¦

Tkernel
Ubias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses*
û
\iter

]beta_1

^beta_2
	_decaym©mªm«m¬*m­+m®Em¯Fm°Tm±Um²v³v´vµv¶*v·+v¸Ev¹FvºTv»Uv¼vhat½vhat¾vhat¿vhatÀ*vhatÁ+vhatÂEvhatÃFvhatÄTvhatÅUvhatÆ*
J
0
1
2
3
*4
+5
E6
F7
T8
U9*
J
0
1
2
3
*4
+5
E6
F7
T8
U9*

`0
a1* 
°
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

gserving_default* 
_Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEconv1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
conv1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
	
`0* 
°
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
ractivity_regularizer_fn
*#&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses* 
* 
* 
\V
VARIABLE_VALUEconv2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
conv2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

*0
+1*

*0
+1*
	
a0* 
°
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
~activity_regularizer_fn
*1&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses* 
* 
* 
* 
]W
VARIABLE_VALUEdense1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdense1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

E0
F1*

E0
F1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses* 
* 
* 
* 
`Z
VARIABLE_VALUEout_dense/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEout_dense/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

T0
U1*

T0
U1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*
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
* 
* 
* 
J
0
1
2
3
4
5
6
7
	8

9*

0
1*
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
	
`0* 
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
	
a0* 
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
* 
<

 total

¡count
¢	variables
£	keras_api*
M

¤total

¥count
¦
_fn_kwargs
§	variables
¨	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

 0
¡1*

¢	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

¤0
¥1*

§	variables*
|
VARIABLE_VALUEAdam/conv2d_3/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_3/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/dense1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/out_dense/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/out_dense/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv2d_3/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_3/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/dense1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/out_dense/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/out_dense/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/conv2d_3/kernel/vhatUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/conv2d_3/bias/vhatSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/conv1/kernel/vhatUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/conv1/bias/vhatSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/conv2/kernel/vhatUlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/conv2/bias/vhatSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/dense1/kernel/vhatUlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense1/bias/vhatSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/out_dense/kernel/vhatUlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/out_dense/bias/vhatSlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_14Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ22
Ý
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_14conv2d_3/kernelconv2d_3/biasconv1/kernel
conv1/biasconv2/kernel
conv2/biasdense1/kerneldense1/biasout_dense/kernelout_dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_465613
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOp conv2/kernel/Read/ReadVariableOpconv2/bias/Read/ReadVariableOp!dense1/kernel/Read/ReadVariableOpdense1/bias/Read/ReadVariableOp$out_dense/kernel/Read/ReadVariableOp"out_dense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp'Adam/conv1/kernel/m/Read/ReadVariableOp%Adam/conv1/bias/m/Read/ReadVariableOp'Adam/conv2/kernel/m/Read/ReadVariableOp%Adam/conv2/bias/m/Read/ReadVariableOp(Adam/dense1/kernel/m/Read/ReadVariableOp&Adam/dense1/bias/m/Read/ReadVariableOp+Adam/out_dense/kernel/m/Read/ReadVariableOp)Adam/out_dense/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp'Adam/conv1/kernel/v/Read/ReadVariableOp%Adam/conv1/bias/v/Read/ReadVariableOp'Adam/conv2/kernel/v/Read/ReadVariableOp%Adam/conv2/bias/v/Read/ReadVariableOp(Adam/dense1/kernel/v/Read/ReadVariableOp&Adam/dense1/bias/v/Read/ReadVariableOp+Adam/out_dense/kernel/v/Read/ReadVariableOp)Adam/out_dense/bias/v/Read/ReadVariableOp-Adam/conv2d_3/kernel/vhat/Read/ReadVariableOp+Adam/conv2d_3/bias/vhat/Read/ReadVariableOp*Adam/conv1/kernel/vhat/Read/ReadVariableOp(Adam/conv1/bias/vhat/Read/ReadVariableOp*Adam/conv2/kernel/vhat/Read/ReadVariableOp(Adam/conv2/bias/vhat/Read/ReadVariableOp+Adam/dense1/kernel/vhat/Read/ReadVariableOp)Adam/dense1/bias/vhat/Read/ReadVariableOp.Adam/out_dense/kernel/vhat/Read/ReadVariableOp,Adam/out_dense/bias/vhat/Read/ReadVariableOpConst*=
Tin6
422	*
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
GPU2*0J 8 *(
f#R!
__inference__traced_save_466032
Ò	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_3/kernelconv2d_3/biasconv1/kernel
conv1/biasconv2/kernel
conv2/biasdense1/kerneldense1/biasout_dense/kernelout_dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcounttotal_1count_1Adam/conv2d_3/kernel/mAdam/conv2d_3/bias/mAdam/conv1/kernel/mAdam/conv1/bias/mAdam/conv2/kernel/mAdam/conv2/bias/mAdam/dense1/kernel/mAdam/dense1/bias/mAdam/out_dense/kernel/mAdam/out_dense/bias/mAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/vAdam/conv1/kernel/vAdam/conv1/bias/vAdam/conv2/kernel/vAdam/conv2/bias/vAdam/dense1/kernel/vAdam/dense1/bias/vAdam/out_dense/kernel/vAdam/out_dense/bias/vAdam/conv2d_3/kernel/vhatAdam/conv2d_3/bias/vhatAdam/conv1/kernel/vhatAdam/conv1/bias/vhatAdam/conv2/kernel/vhatAdam/conv2/bias/vhatAdam/dense1/kernel/vhatAdam/dense1/bias/vhatAdam/out_dense/kernel/vhatAdam/out_dense/bias/vhat*<
Tin5
321*
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
GPU2*0J 8 *+
f&R$
"__inference__traced_restore_466186ÇÄ

±R
í
H__inference_sequential_4_layer_call_and_return_conditional_losses_465274
input_14)
conv2d_3_465213:
conv2d_3_465215:&
conv1_465218:
conv1_465220:'
conv2_465232:
conv2_465234:	!
dense1_465248:
ò@
dense1_465250:@#
out_dense_465254:	@
out_dense_465256:	
identity

identity_1

identity_2¢conv1/StatefulPartitionedCall¢.conv1/kernel/Regularizer/Square/ReadVariableOp¢conv2/StatefulPartitionedCall¢.conv2/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_3/StatefulPartitionedCall¢dense1/StatefulPartitionedCall¢!out_dense/StatefulPartitionedCallý
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinput_14conv2d_3_465213conv2d_3_465215*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ//*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_464773
conv1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv1_465218conv1_465220*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_464796Ä
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
GPU2*0J 8 *6
f1R/
-__inference_conv1_activity_regularizer_464719u
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
valueB:Ó
'conv1/ActivityRegularizer/strided_sliceStridedSlice(conv1/ActivityRegularizer/Shape:output:06conv1/ActivityRegularizer/strided_slice/stack:output:08conv1/ActivityRegularizer/strided_slice/stack_1:output:08conv1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv1/ActivityRegularizer/CastCast0conv1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ¥
!conv1/ActivityRegularizer/truedivRealDiv2conv1/ActivityRegularizer/PartitionedCall:output:0"conv1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: å
max_pool1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_max_pool1_layer_call_and_return_conditional_losses_464728
conv2/StatefulPartitionedCallStatefulPartitionedCall"max_pool1/PartitionedCall:output:0conv2_465232conv2_465234*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_464828Ä
)conv2/ActivityRegularizer/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *6
f1R/
-__inference_conv2_activity_regularizer_464744u
conv2/ActivityRegularizer/ShapeShape&conv2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-conv2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/conv2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/conv2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
'conv2/ActivityRegularizer/strided_sliceStridedSlice(conv2/ActivityRegularizer/Shape:output:06conv2/ActivityRegularizer/strided_slice/stack:output:08conv2/ActivityRegularizer/strided_slice/stack_1:output:08conv2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2/ActivityRegularizer/CastCast0conv2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ¥
!conv2/ActivityRegularizer/truedivRealDiv2conv2/ActivityRegularizer/PartitionedCall:output:0"conv2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: æ
max_pool2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_max_pool2_layer_call_and_return_conditional_losses_464753Ù
flatten1/PartitionedCallPartitionedCall"max_pool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten1_layer_call_and_return_conditional_losses_464849Ø
dropout1/PartitionedCallPartitionedCall!flatten1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout1_layer_call_and_return_conditional_losses_464856
dense1/StatefulPartitionedCallStatefulPartitionedCall!dropout1/PartitionedCall:output:0dense1_465248dense1_465250*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_464869Ü
dropout2/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout2_layer_call_and_return_conditional_losses_464880
!out_dense/StatefulPartitionedCallStatefulPartitionedCall!dropout2/PartitionedCall:output:0out_dense_465254out_dense_465256*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_out_dense_layer_call_and_return_conditional_losses_464893
.conv1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1_465218*&
_output_shapes
:*
dtype0
conv1/kernel/Regularizer/SquareSquare6conv1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:w
conv1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv1/kernel/Regularizer/SumSum#conv1/kernel/Regularizer/Square:y:0'conv1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
conv1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
conv1/kernel/Regularizer/mulMul'conv1/kernel/Regularizer/mul/x:output:0%conv1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.conv2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2_465232*'
_output_shapes
:*
dtype0
conv2/kernel/Regularizer/SquareSquare6conv2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:w
conv2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2/kernel/Regularizer/SumSum#conv2/kernel/Regularizer/Square:y:0'conv2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
conv2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
conv2/kernel/Regularizer/mulMul'conv2/kernel/Regularizer/mul/x:output:0%conv2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: z
IdentityIdentity*out_dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe

Identity_1Identity%conv1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: e

Identity_2Identity%conv2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: Ð
NoOpNoOp^conv1/StatefulPartitionedCall/^conv1/kernel/Regularizer/Square/ReadVariableOp^conv2/StatefulPartitionedCall/^conv2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall^dense1/StatefulPartitionedCall"^out_dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ22: : : : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2`
.conv1/kernel/Regularizer/Square/ReadVariableOp.conv1/kernel/Regularizer/Square/ReadVariableOp2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2`
.conv2/kernel/Regularizer/Square/ReadVariableOp.conv2/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2F
!out_dense/StatefulPartitionedCall!out_dense/StatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"
_user_specified_name
input_14
Ç

'__inference_dense1_layer_call_fn_465751

inputs
unknown:
ò@
	unknown_0:@
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_464869o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿò: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
 
_user_specified_nameinputs
ý
«
A__inference_conv1_layer_call_and_return_conditional_losses_464796

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢.conv1/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..
.conv1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
conv1/kernel/Regularizer/SquareSquare6conv1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:w
conv1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv1/kernel/Regularizer/SumSum#conv1/kernel/Regularizer/Square:y:0'conv1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
conv1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
conv1/kernel/Regularizer/mulMul'conv1/kernel/Regularizer/mul/x:output:0%conv1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..¨
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ//: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv1/kernel/Regularizer/Square/ReadVariableOp.conv1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ//
 
_user_specified_nameinputs
¡
³
__inference_loss_fn_0_465820Q
7conv1_kernel_regularizer_square_readvariableop_resource:
identity¢.conv1/kernel/Regularizer/Square/ReadVariableOp®
.conv1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv1_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype0
conv1/kernel/Regularizer/SquareSquare6conv1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:w
conv1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv1/kernel/Regularizer/SumSum#conv1/kernel/Regularizer/Square:y:0'conv1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
conv1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
conv1/kernel/Regularizer/mulMul'conv1/kernel/Regularizer/mul/x:output:0%conv1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentity conv1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: w
NoOpNoOp/^conv1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.conv1/kernel/Regularizer/Square/ReadVariableOp.conv1/kernel/Regularizer/Square/ReadVariableOp
¯
F
*__inference_max_pool1_layer_call_fn_465663

inputs
identityÖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_max_pool1_layer_call_and_return_conditional_losses_464728
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

­
A__inference_conv2_layer_call_and_return_conditional_losses_465865

inputs9
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢.conv2/kernel/Regularizer/Square/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.conv2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
conv2/kernel/Regularizer/SquareSquare6conv2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:w
conv2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2/kernel/Regularizer/SumSum#conv2/kernel/Regularizer/Square:y:0'conv2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
conv2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
conv2/kernel/Regularizer/mulMul'conv2/kernel/Regularizer/mul/x:output:0%conv2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2/kernel/Regularizer/Square/ReadVariableOp.conv2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú
b
)__inference_dropout1_layer_call_fn_465725

inputs
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout1_layer_call_and_return_conditional_losses_465002q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿò22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
 
_user_specified_nameinputs
ë

&__inference_conv2_layer_call_fn_465683

inputs"
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_464828x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡

õ
B__inference_dense1_layer_call_and_return_conditional_losses_465762

inputs2
matmul_readvariableop_resource:
ò@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ò@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿò: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
 
_user_specified_nameinputs
£


$__inference_signature_wrapper_465613
input_14!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:$
	unknown_3:
	unknown_4:	
	unknown_5:
ò@
	unknown_6:@
	unknown_7:	@
	unknown_8:	
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinput_14unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_464706p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ22: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"
_user_specified_name
input_14
©

ø
E__inference_out_dense_layer_call_and_return_conditional_losses_465809

inputs1
matmul_readvariableop_resource:	@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ño
ù
H__inference_sequential_4_layer_call_and_return_conditional_losses_465586

inputsA
'conv2d_3_conv2d_readvariableop_resource:6
(conv2d_3_biasadd_readvariableop_resource:>
$conv1_conv2d_readvariableop_resource:3
%conv1_biasadd_readvariableop_resource:?
$conv2_conv2d_readvariableop_resource:4
%conv2_biasadd_readvariableop_resource:	9
%dense1_matmul_readvariableop_resource:
ò@4
&dense1_biasadd_readvariableop_resource:@;
(out_dense_matmul_readvariableop_resource:	@8
)out_dense_biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢conv1/BiasAdd/ReadVariableOp¢conv1/Conv2D/ReadVariableOp¢.conv1/kernel/Regularizer/Square/ReadVariableOp¢conv2/BiasAdd/ReadVariableOp¢conv2/Conv2D/ReadVariableOp¢.conv2/kernel/Regularizer/Square/ReadVariableOp¢conv2d_3/BiasAdd/ReadVariableOp¢conv2d_3/Conv2D/ReadVariableOp¢dense1/BiasAdd/ReadVariableOp¢dense1/MatMul/ReadVariableOp¢ out_dense/BiasAdd/ReadVariableOp¢out_dense/MatMul/ReadVariableOp
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¬
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ//*
paddingVALID*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ//
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¹
conv1/Conv2DConv2Dconv2d_3/BiasAdd:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..*
paddingVALID*
strides
~
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..d

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..~
 conv1/ActivityRegularizer/SquareSquareconv1/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..x
conv1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv1/ActivityRegularizer/SumSum$conv1/ActivityRegularizer/Square:y:0(conv1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: d
conv1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
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
valueB:Ó
'conv1/ActivityRegularizer/strided_sliceStridedSlice(conv1/ActivityRegularizer/Shape:output:06conv1/ActivityRegularizer/strided_slice/stack:output:08conv1/ActivityRegularizer/strided_slice/stack_1:output:08conv1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv1/ActivityRegularizer/CastCast0conv1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
!conv1/ActivityRegularizer/truedivRealDiv!conv1/ActivityRegularizer/mul:z:0"conv1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: £
max_pool1/MaxPoolMaxPoolconv1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0»
conv2/Conv2DConv2Dmax_pool1/MaxPool:output:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿe

conv2/ReluReluconv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 conv2/ActivityRegularizer/SquareSquareconv2/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
conv2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2/ActivityRegularizer/SumSum$conv2/ActivityRegularizer/Square:y:0(conv2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: d
conv2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
conv2/ActivityRegularizer/mulMul(conv2/ActivityRegularizer/mul/x:output:0&conv2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: g
conv2/ActivityRegularizer/ShapeShapeconv2/Relu:activations:0*
T0*
_output_shapes
:w
-conv2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/conv2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/conv2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
'conv2/ActivityRegularizer/strided_sliceStridedSlice(conv2/ActivityRegularizer/Shape:output:06conv2/ActivityRegularizer/strided_slice/stack:output:08conv2/ActivityRegularizer/strided_slice/stack_1:output:08conv2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2/ActivityRegularizer/CastCast0conv2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
!conv2/ActivityRegularizer/truedivRealDiv!conv2/ActivityRegularizer/mul:z:0"conv2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ¤
max_pool2/MaxPoolMaxPoolconv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
_
flatten1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ y  
flatten1/ReshapeReshapemax_pool2/MaxPool:output:0flatten1/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò[
dropout1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout1/dropout/MulMulflatten1/Reshape:output:0dropout1/dropout/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò_
dropout1/dropout/ShapeShapeflatten1/Reshape:output:0*
T0*
_output_shapes
: 
-dropout1/dropout/random_uniform/RandomUniformRandomUniformdropout1/dropout/Shape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò*
dtype0d
dropout1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ã
dropout1/dropout/GreaterEqualGreaterEqual6dropout1/dropout/random_uniform/RandomUniform:output:0(dropout1/dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
dropout1/dropout/CastCast!dropout1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
dropout1/dropout/Mul_1Muldropout1/dropout/Mul:z:0dropout1/dropout/Cast:y:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource* 
_output_shapes
:
ò@*
dtype0
dense1/MatMulMatMuldropout1/dropout/Mul_1:z:0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
dense1/ReluReludense1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[
dropout2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout2/dropout/MulMuldense1/Relu:activations:0dropout2/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
dropout2/dropout/ShapeShapedense1/Relu:activations:0*
T0*
_output_shapes
:
-dropout2/dropout/random_uniform/RandomUniformRandomUniformdropout2/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0d
dropout2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Á
dropout2/dropout/GreaterEqualGreaterEqual6dropout2/dropout/random_uniform/RandomUniform:output:0(dropout2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout2/dropout/CastCast!dropout2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout2/dropout/Mul_1Muldropout2/dropout/Mul:z:0dropout2/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
out_dense/MatMul/ReadVariableOpReadVariableOp(out_dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
out_dense/MatMulMatMuldropout2/dropout/Mul_1:z:0'out_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 out_dense/BiasAdd/ReadVariableOpReadVariableOp)out_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
out_dense/BiasAddBiasAddout_dense/MatMul:product:0(out_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
out_dense/SoftmaxSoftmaxout_dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.conv1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
conv1/kernel/Regularizer/SquareSquare6conv1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:w
conv1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv1/kernel/Regularizer/SumSum#conv1/kernel/Regularizer/Square:y:0'conv1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
conv1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
conv1/kernel/Regularizer/mulMul'conv1/kernel/Regularizer/mul/x:output:0%conv1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.conv2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
conv2/kernel/Regularizer/SquareSquare6conv2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:w
conv2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2/kernel/Regularizer/SumSum#conv2/kernel/Regularizer/Square:y:0'conv2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
conv2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
conv2/kernel/Regularizer/mulMul'conv2/kernel/Regularizer/mul/x:output:0%conv2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: k
IdentityIdentityout_dense/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe

Identity_1Identity%conv1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: e

Identity_2Identity%conv2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: é
NoOpNoOp^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp/^conv1/kernel/Regularizer/Square/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp/^conv2/kernel/Regularizer/Square/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp!^out_dense/BiasAdd/ReadVariableOp ^out_dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ22: : : : : : : : : : 2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2`
.conv1/kernel/Regularizer/Square/ReadVariableOp.conv1/kernel/Regularizer/Square/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2`
.conv2/kernel/Regularizer/Square/ReadVariableOp.conv2/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2D
 out_dense/BiasAdd/ReadVariableOp out_dense/BiasAdd/ReadVariableOp2B
out_dense/MatMul/ReadVariableOpout_dense/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
Ù


-__inference_sequential_4_layer_call_fn_464939
input_14!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:$
	unknown_3:
	unknown_4:	
	unknown_5:
ò@
	unknown_6:@
	unknown_7:	@
	unknown_8:	
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinput_14unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: : *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_464914p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ22: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"
_user_specified_name
input_14
¯
F
*__inference_max_pool2_layer_call_fn_465699

inputs
identityÖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_max_pool2_layer_call_and_return_conditional_losses_464753
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

D
-__inference_conv1_activity_regularizer_464719
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
:ÿÿÿÿÿÿÿÿÿG
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
 *¬Å'7I
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
¨

ý
D__inference_conv2d_3_layer_call_and_return_conditional_losses_464773

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ//*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ//g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ//w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
¾
Ì
E__inference_conv2_layer_call_and_return_all_conditional_losses_465694

inputs"
unknown:
	unknown_0:	
identity

identity_1¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_464828¤
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
GPU2*0J 8 *6
f1R/
-__inference_conv2_activity_regularizer_464744x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿX

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
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò	
c
D__inference_dropout2_layer_call_and_return_conditional_losses_464969

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
©

ø
E__inference_out_dense_layer_call_and_return_conditional_losses_464893

inputs1
matmul_readvariableop_resource:	@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
«R
ë
H__inference_sequential_4_layer_call_and_return_conditional_losses_464914

inputs)
conv2d_3_464774:
conv2d_3_464776:&
conv1_464797:
conv1_464799:'
conv2_464829:
conv2_464831:	!
dense1_464870:
ò@
dense1_464872:@#
out_dense_464894:	@
out_dense_464896:	
identity

identity_1

identity_2¢conv1/StatefulPartitionedCall¢.conv1/kernel/Regularizer/Square/ReadVariableOp¢conv2/StatefulPartitionedCall¢.conv2/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_3/StatefulPartitionedCall¢dense1/StatefulPartitionedCall¢!out_dense/StatefulPartitionedCallû
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_464774conv2d_3_464776*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ//*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_464773
conv1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv1_464797conv1_464799*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_464796Ä
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
GPU2*0J 8 *6
f1R/
-__inference_conv1_activity_regularizer_464719u
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
valueB:Ó
'conv1/ActivityRegularizer/strided_sliceStridedSlice(conv1/ActivityRegularizer/Shape:output:06conv1/ActivityRegularizer/strided_slice/stack:output:08conv1/ActivityRegularizer/strided_slice/stack_1:output:08conv1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv1/ActivityRegularizer/CastCast0conv1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ¥
!conv1/ActivityRegularizer/truedivRealDiv2conv1/ActivityRegularizer/PartitionedCall:output:0"conv1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: å
max_pool1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_max_pool1_layer_call_and_return_conditional_losses_464728
conv2/StatefulPartitionedCallStatefulPartitionedCall"max_pool1/PartitionedCall:output:0conv2_464829conv2_464831*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_464828Ä
)conv2/ActivityRegularizer/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *6
f1R/
-__inference_conv2_activity_regularizer_464744u
conv2/ActivityRegularizer/ShapeShape&conv2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-conv2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/conv2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/conv2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
'conv2/ActivityRegularizer/strided_sliceStridedSlice(conv2/ActivityRegularizer/Shape:output:06conv2/ActivityRegularizer/strided_slice/stack:output:08conv2/ActivityRegularizer/strided_slice/stack_1:output:08conv2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2/ActivityRegularizer/CastCast0conv2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ¥
!conv2/ActivityRegularizer/truedivRealDiv2conv2/ActivityRegularizer/PartitionedCall:output:0"conv2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: æ
max_pool2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_max_pool2_layer_call_and_return_conditional_losses_464753Ù
flatten1/PartitionedCallPartitionedCall"max_pool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten1_layer_call_and_return_conditional_losses_464849Ø
dropout1/PartitionedCallPartitionedCall!flatten1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout1_layer_call_and_return_conditional_losses_464856
dense1/StatefulPartitionedCallStatefulPartitionedCall!dropout1/PartitionedCall:output:0dense1_464870dense1_464872*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_464869Ü
dropout2/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout2_layer_call_and_return_conditional_losses_464880
!out_dense/StatefulPartitionedCallStatefulPartitionedCall!dropout2/PartitionedCall:output:0out_dense_464894out_dense_464896*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_out_dense_layer_call_and_return_conditional_losses_464893
.conv1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1_464797*&
_output_shapes
:*
dtype0
conv1/kernel/Regularizer/SquareSquare6conv1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:w
conv1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv1/kernel/Regularizer/SumSum#conv1/kernel/Regularizer/Square:y:0'conv1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
conv1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
conv1/kernel/Regularizer/mulMul'conv1/kernel/Regularizer/mul/x:output:0%conv1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.conv2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2_464829*'
_output_shapes
:*
dtype0
conv2/kernel/Regularizer/SquareSquare6conv2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:w
conv2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2/kernel/Regularizer/SumSum#conv2/kernel/Regularizer/Square:y:0'conv2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
conv2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
conv2/kernel/Regularizer/mulMul'conv2/kernel/Regularizer/mul/x:output:0%conv2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: z
IdentityIdentity*out_dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe

Identity_1Identity%conv1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: e

Identity_2Identity%conv2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: Ð
NoOpNoOp^conv1/StatefulPartitionedCall/^conv1/kernel/Regularizer/Square/ReadVariableOp^conv2/StatefulPartitionedCall/^conv2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall^dense1/StatefulPartitionedCall"^out_dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ22: : : : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2`
.conv1/kernel/Regularizer/Square/ReadVariableOp.conv1/kernel/Regularizer/Square/ReadVariableOp2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2`
.conv2/kernel/Regularizer/Square/ReadVariableOp.conv2/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2F
!out_dense/StatefulPartitionedCall!out_dense/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
Ë

*__inference_out_dense_layer_call_fn_465798

inputs
unknown:	@
	unknown_0:	
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_out_dense_layer_call_and_return_conditional_losses_464893p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ê
`
D__inference_flatten1_layer_call_and_return_conditional_losses_464849

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ y  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿòZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í

)__inference_conv2d_3_layer_call_fn_465622

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ//*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_464773w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ//`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ22: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
Ù


-__inference_sequential_4_layer_call_fn_465210
input_14!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:$
	unknown_3:
	unknown_4:	
	unknown_5:
ò@
	unknown_6:@
	unknown_7:	@
	unknown_8:	
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinput_14unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: : *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_465158p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ22: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"
_user_specified_name
input_14
U
±
H__inference_sequential_4_layer_call_and_return_conditional_losses_465158

inputs)
conv2d_3_465097:
conv2d_3_465099:&
conv1_465102:
conv1_465104:'
conv2_465116:
conv2_465118:	!
dense1_465132:
ò@
dense1_465134:@#
out_dense_465138:	@
out_dense_465140:	
identity

identity_1

identity_2¢conv1/StatefulPartitionedCall¢.conv1/kernel/Regularizer/Square/ReadVariableOp¢conv2/StatefulPartitionedCall¢.conv2/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_3/StatefulPartitionedCall¢dense1/StatefulPartitionedCall¢ dropout1/StatefulPartitionedCall¢ dropout2/StatefulPartitionedCall¢!out_dense/StatefulPartitionedCallû
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_465097conv2d_3_465099*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ//*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_464773
conv1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv1_465102conv1_465104*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_464796Ä
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
GPU2*0J 8 *6
f1R/
-__inference_conv1_activity_regularizer_464719u
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
valueB:Ó
'conv1/ActivityRegularizer/strided_sliceStridedSlice(conv1/ActivityRegularizer/Shape:output:06conv1/ActivityRegularizer/strided_slice/stack:output:08conv1/ActivityRegularizer/strided_slice/stack_1:output:08conv1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv1/ActivityRegularizer/CastCast0conv1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ¥
!conv1/ActivityRegularizer/truedivRealDiv2conv1/ActivityRegularizer/PartitionedCall:output:0"conv1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: å
max_pool1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_max_pool1_layer_call_and_return_conditional_losses_464728
conv2/StatefulPartitionedCallStatefulPartitionedCall"max_pool1/PartitionedCall:output:0conv2_465116conv2_465118*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_464828Ä
)conv2/ActivityRegularizer/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *6
f1R/
-__inference_conv2_activity_regularizer_464744u
conv2/ActivityRegularizer/ShapeShape&conv2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-conv2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/conv2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/conv2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
'conv2/ActivityRegularizer/strided_sliceStridedSlice(conv2/ActivityRegularizer/Shape:output:06conv2/ActivityRegularizer/strided_slice/stack:output:08conv2/ActivityRegularizer/strided_slice/stack_1:output:08conv2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2/ActivityRegularizer/CastCast0conv2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ¥
!conv2/ActivityRegularizer/truedivRealDiv2conv2/ActivityRegularizer/PartitionedCall:output:0"conv2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: æ
max_pool2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_max_pool2_layer_call_and_return_conditional_losses_464753Ù
flatten1/PartitionedCallPartitionedCall"max_pool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten1_layer_call_and_return_conditional_losses_464849è
 dropout1/StatefulPartitionedCallStatefulPartitionedCall!flatten1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout1_layer_call_and_return_conditional_losses_465002
dense1/StatefulPartitionedCallStatefulPartitionedCall)dropout1/StatefulPartitionedCall:output:0dense1_465132dense1_465134*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_464869
 dropout2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0!^dropout1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout2_layer_call_and_return_conditional_losses_464969
!out_dense/StatefulPartitionedCallStatefulPartitionedCall)dropout2/StatefulPartitionedCall:output:0out_dense_465138out_dense_465140*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_out_dense_layer_call_and_return_conditional_losses_464893
.conv1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1_465102*&
_output_shapes
:*
dtype0
conv1/kernel/Regularizer/SquareSquare6conv1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:w
conv1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv1/kernel/Regularizer/SumSum#conv1/kernel/Regularizer/Square:y:0'conv1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
conv1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
conv1/kernel/Regularizer/mulMul'conv1/kernel/Regularizer/mul/x:output:0%conv1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.conv2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2_465116*'
_output_shapes
:*
dtype0
conv2/kernel/Regularizer/SquareSquare6conv2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:w
conv2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2/kernel/Regularizer/SumSum#conv2/kernel/Regularizer/Square:y:0'conv2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
conv2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
conv2/kernel/Regularizer/mulMul'conv2/kernel/Regularizer/mul/x:output:0%conv2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: z
IdentityIdentity*out_dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe

Identity_1Identity%conv1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: e

Identity_2Identity%conv2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp^conv1/StatefulPartitionedCall/^conv1/kernel/Regularizer/Square/ReadVariableOp^conv2/StatefulPartitionedCall/^conv2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall^dense1/StatefulPartitionedCall!^dropout1/StatefulPartitionedCall!^dropout2/StatefulPartitionedCall"^out_dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ22: : : : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2`
.conv1/kernel/Regularizer/Square/ReadVariableOp.conv1/kernel/Regularizer/Square/ReadVariableOp2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2`
.conv2/kernel/Regularizer/Square/ReadVariableOp.conv2/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2D
 dropout1/StatefulPartitionedCall dropout1/StatefulPartitionedCall2D
 dropout2/StatefulPartitionedCall dropout2/StatefulPartitionedCall2F
!out_dense/StatefulPartitionedCall!out_dense/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
¡

õ
B__inference_dense1_layer_call_and_return_conditional_losses_464869

inputs2
matmul_readvariableop_resource:
ò@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ò@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿò: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
 
_user_specified_nameinputs

a
E__inference_max_pool1_layer_call_and_return_conditional_losses_465668

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


c
D__inference_dropout1_layer_call_and_return_conditional_losses_465742

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
dropout/MulMulinputsdropout/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿòC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¨
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿòq
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿòk
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò[
IdentityIdentitydropout/Mul_1:z:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿò:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
 
_user_specified_nameinputs
¨

ý
D__inference_conv2d_3_layer_call_and_return_conditional_losses_465632

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ//*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ//g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ//w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
 
E
)__inference_dropout2_layer_call_fn_465767

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout2_layer_call_and_return_conditional_losses_464880`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
º
Ê
E__inference_conv1_layer_call_and_return_all_conditional_losses_465658

inputs!
unknown:
	unknown_0:
identity

identity_1¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_464796¤
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
GPU2*0J 8 *6
f1R/
-__inference_conv1_activity_regularizer_464719w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..X

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
:ÿÿÿÿÿÿÿÿÿ//: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ//
 
_user_specified_nameinputs
¨
E
)__inference_dropout1_layer_call_fn_465720

inputs
identity´
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout1_layer_call_and_return_conditional_losses_464856b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿò:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
 
_user_specified_nameinputs
ç

&__inference_conv1_layer_call_fn_465647

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_464796w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ//: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ//
 
_user_specified_nameinputs
Ê
`
D__inference_flatten1_layer_call_and_return_conditional_losses_465715

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ y  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿòZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

D
-__inference_conv2_activity_regularizer_464744
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
:ÿÿÿÿÿÿÿÿÿG
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
 *¬Å'7I
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
×
b
D__inference_dropout2_layer_call_and_return_conditional_losses_464880

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

a
E__inference_max_pool2_layer_call_and_return_conditional_losses_465704

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò
b
)__inference_dropout2_layer_call_fn_465772

inputs
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout2_layer_call_and_return_conditional_losses_464969o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ý
«
A__inference_conv1_layer_call_and_return_conditional_losses_465848

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢.conv1/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..
.conv1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
conv1/kernel/Regularizer/SquareSquare6conv1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:w
conv1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv1/kernel/Regularizer/SumSum#conv1/kernel/Regularizer/Square:y:0'conv1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
conv1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
conv1/kernel/Regularizer/mulMul'conv1/kernel/Regularizer/mul/x:output:0%conv1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..¨
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ//: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv1/kernel/Regularizer/Square/ReadVariableOp.conv1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ//
 
_user_specified_nameinputs
ïÀ
â
"__inference__traced_restore_466186
file_prefix:
 assignvariableop_conv2d_3_kernel:.
 assignvariableop_1_conv2d_3_bias:9
assignvariableop_2_conv1_kernel:+
assignvariableop_3_conv1_bias::
assignvariableop_4_conv2_kernel:,
assignvariableop_5_conv2_bias:	4
 assignvariableop_6_dense1_kernel:
ò@,
assignvariableop_7_dense1_bias:@6
#assignvariableop_8_out_dense_kernel:	@0
!assignvariableop_9_out_dense_bias:	'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: #
assignvariableop_14_total: #
assignvariableop_15_count: %
assignvariableop_16_total_1: %
assignvariableop_17_count_1: D
*assignvariableop_18_adam_conv2d_3_kernel_m:6
(assignvariableop_19_adam_conv2d_3_bias_m:A
'assignvariableop_20_adam_conv1_kernel_m:3
%assignvariableop_21_adam_conv1_bias_m:B
'assignvariableop_22_adam_conv2_kernel_m:4
%assignvariableop_23_adam_conv2_bias_m:	<
(assignvariableop_24_adam_dense1_kernel_m:
ò@4
&assignvariableop_25_adam_dense1_bias_m:@>
+assignvariableop_26_adam_out_dense_kernel_m:	@8
)assignvariableop_27_adam_out_dense_bias_m:	D
*assignvariableop_28_adam_conv2d_3_kernel_v:6
(assignvariableop_29_adam_conv2d_3_bias_v:A
'assignvariableop_30_adam_conv1_kernel_v:3
%assignvariableop_31_adam_conv1_bias_v:B
'assignvariableop_32_adam_conv2_kernel_v:4
%assignvariableop_33_adam_conv2_bias_v:	<
(assignvariableop_34_adam_dense1_kernel_v:
ò@4
&assignvariableop_35_adam_dense1_bias_v:@>
+assignvariableop_36_adam_out_dense_kernel_v:	@8
)assignvariableop_37_adam_out_dense_bias_v:	G
-assignvariableop_38_adam_conv2d_3_kernel_vhat:9
+assignvariableop_39_adam_conv2d_3_bias_vhat:D
*assignvariableop_40_adam_conv1_kernel_vhat:6
(assignvariableop_41_adam_conv1_bias_vhat:E
*assignvariableop_42_adam_conv2_kernel_vhat:7
(assignvariableop_43_adam_conv2_bias_vhat:	?
+assignvariableop_44_adam_dense1_kernel_vhat:
ò@7
)assignvariableop_45_adam_dense1_bias_vhat:@A
.assignvariableop_46_adam_out_dense_kernel_vhat:	@;
,assignvariableop_47_adam_out_dense_bias_vhat:	
identity_49¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*º
value°B­1B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÒ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*u
valuelBj1B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ú
_output_shapesÇ
Ä:::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes5
321	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_conv2d_3_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_3_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_conv2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp#assignvariableop_8_out_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp!assignvariableop_9_out_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_conv2d_3_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_conv2d_3_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_conv1_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp%assignvariableop_21_adam_conv1_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_conv2_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp%assignvariableop_23_adam_conv2_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense1_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp&assignvariableop_25_adam_dense1_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp+assignvariableop_26_adam_out_dense_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_out_dense_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_conv2d_3_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_conv2d_3_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_conv1_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp%assignvariableop_31_adam_conv1_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_conv2_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp%assignvariableop_33_adam_conv2_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense1_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp&assignvariableop_35_adam_dense1_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp+assignvariableop_36_adam_out_dense_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_out_dense_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp-assignvariableop_38_adam_conv2d_3_kernel_vhatIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_3_bias_vhatIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_conv1_kernel_vhatIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_conv1_bias_vhatIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_conv2_kernel_vhatIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_conv2_bias_vhatIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp+assignvariableop_44_adam_dense1_kernel_vhatIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense1_bias_vhatIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp.assignvariableop_46_adam_out_dense_kernel_vhatIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp,assignvariableop_47_adam_out_dense_bias_vhatIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ï
Identity_48Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_49IdentityIdentity_48:output:0^NoOp_1*
T0*
_output_shapes
: Ü
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_49Identity_49:output:0*u
_input_shapesd
b: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472(
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
ò	
c
D__inference_dropout2_layer_call_and_return_conditional_losses_465789

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Û_
Ö	
!__inference__wrapped_model_464706
input_14N
4sequential_4_conv2d_3_conv2d_readvariableop_resource:C
5sequential_4_conv2d_3_biasadd_readvariableop_resource:K
1sequential_4_conv1_conv2d_readvariableop_resource:@
2sequential_4_conv1_biasadd_readvariableop_resource:L
1sequential_4_conv2_conv2d_readvariableop_resource:A
2sequential_4_conv2_biasadd_readvariableop_resource:	F
2sequential_4_dense1_matmul_readvariableop_resource:
ò@A
3sequential_4_dense1_biasadd_readvariableop_resource:@H
5sequential_4_out_dense_matmul_readvariableop_resource:	@E
6sequential_4_out_dense_biasadd_readvariableop_resource:	
identity¢)sequential_4/conv1/BiasAdd/ReadVariableOp¢(sequential_4/conv1/Conv2D/ReadVariableOp¢)sequential_4/conv2/BiasAdd/ReadVariableOp¢(sequential_4/conv2/Conv2D/ReadVariableOp¢,sequential_4/conv2d_3/BiasAdd/ReadVariableOp¢+sequential_4/conv2d_3/Conv2D/ReadVariableOp¢*sequential_4/dense1/BiasAdd/ReadVariableOp¢)sequential_4/dense1/MatMul/ReadVariableOp¢-sequential_4/out_dense/BiasAdd/ReadVariableOp¢,sequential_4/out_dense/MatMul/ReadVariableOp¨
+sequential_4/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4sequential_4_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0È
sequential_4/conv2d_3/Conv2DConv2Dinput_143sequential_4/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ//*
paddingVALID*
strides

,sequential_4/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¿
sequential_4/conv2d_3/BiasAddBiasAdd%sequential_4/conv2d_3/Conv2D:output:04sequential_4/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ//¢
(sequential_4/conv1/Conv2D/ReadVariableOpReadVariableOp1sequential_4_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0à
sequential_4/conv1/Conv2DConv2D&sequential_4/conv2d_3/BiasAdd:output:00sequential_4/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..*
paddingVALID*
strides

)sequential_4/conv1/BiasAdd/ReadVariableOpReadVariableOp2sequential_4_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¶
sequential_4/conv1/BiasAddBiasAdd"sequential_4/conv1/Conv2D:output:01sequential_4/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..~
sequential_4/conv1/ReluRelu#sequential_4/conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..
-sequential_4/conv1/ActivityRegularizer/SquareSquare%sequential_4/conv1/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..
,sequential_4/conv1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¼
*sequential_4/conv1/ActivityRegularizer/SumSum1sequential_4/conv1/ActivityRegularizer/Square:y:05sequential_4/conv1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: q
,sequential_4/conv1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¾
*sequential_4/conv1/ActivityRegularizer/mulMul5sequential_4/conv1/ActivityRegularizer/mul/x:output:03sequential_4/conv1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 
,sequential_4/conv1/ActivityRegularizer/ShapeShape%sequential_4/conv1/Relu:activations:0*
T0*
_output_shapes
:
:sequential_4/conv1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
<sequential_4/conv1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
<sequential_4/conv1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4sequential_4/conv1/ActivityRegularizer/strided_sliceStridedSlice5sequential_4/conv1/ActivityRegularizer/Shape:output:0Csequential_4/conv1/ActivityRegularizer/strided_slice/stack:output:0Esequential_4/conv1/ActivityRegularizer/strided_slice/stack_1:output:0Esequential_4/conv1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¢
+sequential_4/conv1/ActivityRegularizer/CastCast=sequential_4/conv1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: »
.sequential_4/conv1/ActivityRegularizer/truedivRealDiv.sequential_4/conv1/ActivityRegularizer/mul:z:0/sequential_4/conv1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ½
sequential_4/max_pool1/MaxPoolMaxPool%sequential_4/conv1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
£
(sequential_4/conv2/Conv2D/ReadVariableOpReadVariableOp1sequential_4_conv2_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0â
sequential_4/conv2/Conv2DConv2D'sequential_4/max_pool1/MaxPool:output:00sequential_4/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

)sequential_4/conv2/BiasAdd/ReadVariableOpReadVariableOp2sequential_4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
sequential_4/conv2/BiasAddBiasAdd"sequential_4/conv2/Conv2D:output:01sequential_4/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_4/conv2/ReluRelu#sequential_4/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-sequential_4/conv2/ActivityRegularizer/SquareSquare%sequential_4/conv2/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_4/conv2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ¼
*sequential_4/conv2/ActivityRegularizer/SumSum1sequential_4/conv2/ActivityRegularizer/Square:y:05sequential_4/conv2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: q
,sequential_4/conv2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7¾
*sequential_4/conv2/ActivityRegularizer/mulMul5sequential_4/conv2/ActivityRegularizer/mul/x:output:03sequential_4/conv2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 
,sequential_4/conv2/ActivityRegularizer/ShapeShape%sequential_4/conv2/Relu:activations:0*
T0*
_output_shapes
:
:sequential_4/conv2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
<sequential_4/conv2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
<sequential_4/conv2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4sequential_4/conv2/ActivityRegularizer/strided_sliceStridedSlice5sequential_4/conv2/ActivityRegularizer/Shape:output:0Csequential_4/conv2/ActivityRegularizer/strided_slice/stack:output:0Esequential_4/conv2/ActivityRegularizer/strided_slice/stack_1:output:0Esequential_4/conv2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¢
+sequential_4/conv2/ActivityRegularizer/CastCast=sequential_4/conv2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: »
.sequential_4/conv2/ActivityRegularizer/truedivRealDiv.sequential_4/conv2/ActivityRegularizer/mul:z:0/sequential_4/conv2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ¾
sequential_4/max_pool2/MaxPoolMaxPool%sequential_4/conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
l
sequential_4/flatten1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ y  «
sequential_4/flatten1/ReshapeReshape'sequential_4/max_pool2/MaxPool:output:0$sequential_4/flatten1/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
sequential_4/dropout1/IdentityIdentity&sequential_4/flatten1/Reshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
)sequential_4/dense1/MatMul/ReadVariableOpReadVariableOp2sequential_4_dense1_matmul_readvariableop_resource* 
_output_shapes
:
ò@*
dtype0²
sequential_4/dense1/MatMulMatMul'sequential_4/dropout1/Identity:output:01sequential_4/dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*sequential_4/dense1/BiasAdd/ReadVariableOpReadVariableOp3sequential_4_dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0²
sequential_4/dense1/BiasAddBiasAdd$sequential_4/dense1/MatMul:product:02sequential_4/dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
sequential_4/dense1/ReluRelu$sequential_4/dense1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
sequential_4/dropout2/IdentityIdentity&sequential_4/dense1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@£
,sequential_4/out_dense/MatMul/ReadVariableOpReadVariableOp5sequential_4_out_dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0¹
sequential_4/out_dense/MatMulMatMul'sequential_4/dropout2/Identity:output:04sequential_4/out_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
-sequential_4/out_dense/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_out_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¼
sequential_4/out_dense/BiasAddBiasAdd'sequential_4/out_dense/MatMul:product:05sequential_4/out_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_4/out_dense/SoftmaxSoftmax'sequential_4/out_dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
IdentityIdentity(sequential_4/out_dense/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp*^sequential_4/conv1/BiasAdd/ReadVariableOp)^sequential_4/conv1/Conv2D/ReadVariableOp*^sequential_4/conv2/BiasAdd/ReadVariableOp)^sequential_4/conv2/Conv2D/ReadVariableOp-^sequential_4/conv2d_3/BiasAdd/ReadVariableOp,^sequential_4/conv2d_3/Conv2D/ReadVariableOp+^sequential_4/dense1/BiasAdd/ReadVariableOp*^sequential_4/dense1/MatMul/ReadVariableOp.^sequential_4/out_dense/BiasAdd/ReadVariableOp-^sequential_4/out_dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ22: : : : : : : : : : 2V
)sequential_4/conv1/BiasAdd/ReadVariableOp)sequential_4/conv1/BiasAdd/ReadVariableOp2T
(sequential_4/conv1/Conv2D/ReadVariableOp(sequential_4/conv1/Conv2D/ReadVariableOp2V
)sequential_4/conv2/BiasAdd/ReadVariableOp)sequential_4/conv2/BiasAdd/ReadVariableOp2T
(sequential_4/conv2/Conv2D/ReadVariableOp(sequential_4/conv2/Conv2D/ReadVariableOp2\
,sequential_4/conv2d_3/BiasAdd/ReadVariableOp,sequential_4/conv2d_3/BiasAdd/ReadVariableOp2Z
+sequential_4/conv2d_3/Conv2D/ReadVariableOp+sequential_4/conv2d_3/Conv2D/ReadVariableOp2X
*sequential_4/dense1/BiasAdd/ReadVariableOp*sequential_4/dense1/BiasAdd/ReadVariableOp2V
)sequential_4/dense1/MatMul/ReadVariableOp)sequential_4/dense1/MatMul/ReadVariableOp2^
-sequential_4/out_dense/BiasAdd/ReadVariableOp-sequential_4/out_dense/BiasAdd/ReadVariableOp2\
,sequential_4/out_dense/MatMul/ReadVariableOp,sequential_4/out_dense/MatMul/ReadVariableOp:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"
_user_specified_name
input_14
¤
´
__inference_loss_fn_1_465831R
7conv2_kernel_regularizer_square_readvariableop_resource:
identity¢.conv2/kernel/Regularizer/Square/ReadVariableOp¯
.conv2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv2_kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:*
dtype0
conv2/kernel/Regularizer/SquareSquare6conv2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:w
conv2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2/kernel/Regularizer/SumSum#conv2/kernel/Regularizer/Square:y:0'conv2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
conv2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
conv2/kernel/Regularizer/mulMul'conv2/kernel/Regularizer/mul/x:output:0%conv2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentity conv2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: w
NoOpNoOp/^conv2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.conv2/kernel/Regularizer/Square/ReadVariableOp.conv2/kernel/Regularizer/Square/ReadVariableOp
a
ù
H__inference_sequential_4_layer_call_and_return_conditional_losses_465490

inputsA
'conv2d_3_conv2d_readvariableop_resource:6
(conv2d_3_biasadd_readvariableop_resource:>
$conv1_conv2d_readvariableop_resource:3
%conv1_biasadd_readvariableop_resource:?
$conv2_conv2d_readvariableop_resource:4
%conv2_biasadd_readvariableop_resource:	9
%dense1_matmul_readvariableop_resource:
ò@4
&dense1_biasadd_readvariableop_resource:@;
(out_dense_matmul_readvariableop_resource:	@8
)out_dense_biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢conv1/BiasAdd/ReadVariableOp¢conv1/Conv2D/ReadVariableOp¢.conv1/kernel/Regularizer/Square/ReadVariableOp¢conv2/BiasAdd/ReadVariableOp¢conv2/Conv2D/ReadVariableOp¢.conv2/kernel/Regularizer/Square/ReadVariableOp¢conv2d_3/BiasAdd/ReadVariableOp¢conv2d_3/Conv2D/ReadVariableOp¢dense1/BiasAdd/ReadVariableOp¢dense1/MatMul/ReadVariableOp¢ out_dense/BiasAdd/ReadVariableOp¢out_dense/MatMul/ReadVariableOp
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¬
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ//*
paddingVALID*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ//
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¹
conv1/Conv2DConv2Dconv2d_3/BiasAdd:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..*
paddingVALID*
strides
~
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..d

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..~
 conv1/ActivityRegularizer/SquareSquareconv1/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..x
conv1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv1/ActivityRegularizer/SumSum$conv1/ActivityRegularizer/Square:y:0(conv1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: d
conv1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
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
valueB:Ó
'conv1/ActivityRegularizer/strided_sliceStridedSlice(conv1/ActivityRegularizer/Shape:output:06conv1/ActivityRegularizer/strided_slice/stack:output:08conv1/ActivityRegularizer/strided_slice/stack_1:output:08conv1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv1/ActivityRegularizer/CastCast0conv1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
!conv1/ActivityRegularizer/truedivRealDiv!conv1/ActivityRegularizer/mul:z:0"conv1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: £
max_pool1/MaxPoolMaxPoolconv1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0»
conv2/Conv2DConv2Dmax_pool1/MaxPool:output:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿe

conv2/ReluReluconv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 conv2/ActivityRegularizer/SquareSquareconv2/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
conv2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2/ActivityRegularizer/SumSum$conv2/ActivityRegularizer/Square:y:0(conv2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: d
conv2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
conv2/ActivityRegularizer/mulMul(conv2/ActivityRegularizer/mul/x:output:0&conv2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: g
conv2/ActivityRegularizer/ShapeShapeconv2/Relu:activations:0*
T0*
_output_shapes
:w
-conv2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/conv2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/conv2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
'conv2/ActivityRegularizer/strided_sliceStridedSlice(conv2/ActivityRegularizer/Shape:output:06conv2/ActivityRegularizer/strided_slice/stack:output:08conv2/ActivityRegularizer/strided_slice/stack_1:output:08conv2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2/ActivityRegularizer/CastCast0conv2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
!conv2/ActivityRegularizer/truedivRealDiv!conv2/ActivityRegularizer/mul:z:0"conv2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ¤
max_pool2/MaxPoolMaxPoolconv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
_
flatten1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ y  
flatten1/ReshapeReshapemax_pool2/MaxPool:output:0flatten1/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿòl
dropout1/IdentityIdentityflatten1/Reshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource* 
_output_shapes
:
ò@*
dtype0
dense1/MatMulMatMuldropout1/Identity:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
dense1/ReluReludense1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@j
dropout2/IdentityIdentitydense1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
out_dense/MatMul/ReadVariableOpReadVariableOp(out_dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
out_dense/MatMulMatMuldropout2/Identity:output:0'out_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 out_dense/BiasAdd/ReadVariableOpReadVariableOp)out_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
out_dense/BiasAddBiasAddout_dense/MatMul:product:0(out_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
out_dense/SoftmaxSoftmaxout_dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.conv1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
conv1/kernel/Regularizer/SquareSquare6conv1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:w
conv1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv1/kernel/Regularizer/SumSum#conv1/kernel/Regularizer/Square:y:0'conv1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
conv1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
conv1/kernel/Regularizer/mulMul'conv1/kernel/Regularizer/mul/x:output:0%conv1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.conv2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
conv2/kernel/Regularizer/SquareSquare6conv2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:w
conv2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2/kernel/Regularizer/SumSum#conv2/kernel/Regularizer/Square:y:0'conv2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
conv2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
conv2/kernel/Regularizer/mulMul'conv2/kernel/Regularizer/mul/x:output:0%conv2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: k
IdentityIdentityout_dense/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe

Identity_1Identity%conv1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: e

Identity_2Identity%conv2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: é
NoOpNoOp^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp/^conv1/kernel/Regularizer/Square/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp/^conv2/kernel/Regularizer/Square/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp!^out_dense/BiasAdd/ReadVariableOp ^out_dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ22: : : : : : : : : : 2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2`
.conv1/kernel/Regularizer/Square/ReadVariableOp.conv1/kernel/Regularizer/Square/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2`
.conv2/kernel/Regularizer/Square/ReadVariableOp.conv2/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2D
 out_dense/BiasAdd/ReadVariableOp out_dense/BiasAdd/ReadVariableOp2B
out_dense/MatMul/ReadVariableOpout_dense/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs

­
A__inference_conv2_layer_call_and_return_conditional_losses_464828

inputs9
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢.conv2/kernel/Regularizer/Square/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.conv2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
conv2/kernel/Regularizer/SquareSquare6conv2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:w
conv2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2/kernel/Regularizer/SumSum#conv2/kernel/Regularizer/Square:y:0'conv2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
conv2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
conv2/kernel/Regularizer/mulMul'conv2/kernel/Regularizer/mul/x:output:0%conv2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2/kernel/Regularizer/Square/ReadVariableOp.conv2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
b
D__inference_dropout1_layer_call_and_return_conditional_losses_465730

inputs

identity_1P
IdentityIdentityinputs*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò]

Identity_1IdentityIdentity:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿò:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
 
_user_specified_nameinputs
ïb
Å
__inference__traced_save_466032
file_prefix.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop+
'savev2_conv2_kernel_read_readvariableop)
%savev2_conv2_bias_read_readvariableop,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop/
+savev2_out_dense_kernel_read_readvariableop-
)savev2_out_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop2
.savev2_adam_conv1_kernel_m_read_readvariableop0
,savev2_adam_conv1_bias_m_read_readvariableop2
.savev2_adam_conv2_kernel_m_read_readvariableop0
,savev2_adam_conv2_bias_m_read_readvariableop3
/savev2_adam_dense1_kernel_m_read_readvariableop1
-savev2_adam_dense1_bias_m_read_readvariableop6
2savev2_adam_out_dense_kernel_m_read_readvariableop4
0savev2_adam_out_dense_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop2
.savev2_adam_conv1_kernel_v_read_readvariableop0
,savev2_adam_conv1_bias_v_read_readvariableop2
.savev2_adam_conv2_kernel_v_read_readvariableop0
,savev2_adam_conv2_bias_v_read_readvariableop3
/savev2_adam_dense1_kernel_v_read_readvariableop1
-savev2_adam_dense1_bias_v_read_readvariableop6
2savev2_adam_out_dense_kernel_v_read_readvariableop4
0savev2_adam_out_dense_bias_v_read_readvariableop8
4savev2_adam_conv2d_3_kernel_vhat_read_readvariableop6
2savev2_adam_conv2d_3_bias_vhat_read_readvariableop5
1savev2_adam_conv1_kernel_vhat_read_readvariableop3
/savev2_adam_conv1_bias_vhat_read_readvariableop5
1savev2_adam_conv2_kernel_vhat_read_readvariableop3
/savev2_adam_conv2_bias_vhat_read_readvariableop6
2savev2_adam_dense1_kernel_vhat_read_readvariableop4
0savev2_adam_dense1_bias_vhat_read_readvariableop9
5savev2_adam_out_dense_kernel_vhat_read_readvariableop7
3savev2_adam_out_dense_bias_vhat_read_readvariableop
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
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*º
value°B­1B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÏ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*u
valuelBj1B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ù
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableop+savev2_out_dense_kernel_read_readvariableop)savev2_out_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop.savev2_adam_conv1_kernel_m_read_readvariableop,savev2_adam_conv1_bias_m_read_readvariableop.savev2_adam_conv2_kernel_m_read_readvariableop,savev2_adam_conv2_bias_m_read_readvariableop/savev2_adam_dense1_kernel_m_read_readvariableop-savev2_adam_dense1_bias_m_read_readvariableop2savev2_adam_out_dense_kernel_m_read_readvariableop0savev2_adam_out_dense_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop.savev2_adam_conv1_kernel_v_read_readvariableop,savev2_adam_conv1_bias_v_read_readvariableop.savev2_adam_conv2_kernel_v_read_readvariableop,savev2_adam_conv2_bias_v_read_readvariableop/savev2_adam_dense1_kernel_v_read_readvariableop-savev2_adam_dense1_bias_v_read_readvariableop2savev2_adam_out_dense_kernel_v_read_readvariableop0savev2_adam_out_dense_bias_v_read_readvariableop4savev2_adam_conv2d_3_kernel_vhat_read_readvariableop2savev2_adam_conv2d_3_bias_vhat_read_readvariableop1savev2_adam_conv1_kernel_vhat_read_readvariableop/savev2_adam_conv1_bias_vhat_read_readvariableop1savev2_adam_conv2_kernel_vhat_read_readvariableop/savev2_adam_conv2_bias_vhat_read_readvariableop2savev2_adam_dense1_kernel_vhat_read_readvariableop0savev2_adam_dense1_bias_vhat_read_readvariableop5savev2_adam_out_dense_kernel_vhat_read_readvariableop3savev2_adam_out_dense_bias_vhat_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes5
321	
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

identity_1Identity_1:output:0*á
_input_shapesÏ
Ì: :::::::
ò@:@:	@:: : : : : : : : :::::::
ò@:@:	@::::::::
ò@:@:	@::::::::
ò@:@:	@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::-)
'
_output_shapes
::!

_output_shapes	
::&"
 
_output_shapes
:
ò@: 

_output_shapes
:@:%	!

_output_shapes
:	@:!


_output_shapes	
::
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::-)
'
_output_shapes
::!

_output_shapes	
::&"
 
_output_shapes
:
ò@: 

_output_shapes
:@:%!

_output_shapes
:	@:!

_output_shapes	
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::  

_output_shapes
::-!)
'
_output_shapes
::!"

_output_shapes	
::&#"
 
_output_shapes
:
ò@: $

_output_shapes
:@:%%!

_output_shapes
:	@:!&

_output_shapes	
::,'(
&
_output_shapes
:: (

_output_shapes
::,)(
&
_output_shapes
:: *

_output_shapes
::-+)
'
_output_shapes
::!,

_output_shapes	
::&-"
 
_output_shapes
:
ò@: .

_output_shapes
:@:%/!

_output_shapes
:	@:!0

_output_shapes	
::1

_output_shapes
: 
U
³
H__inference_sequential_4_layer_call_and_return_conditional_losses_465338
input_14)
conv2d_3_465277:
conv2d_3_465279:&
conv1_465282:
conv1_465284:'
conv2_465296:
conv2_465298:	!
dense1_465312:
ò@
dense1_465314:@#
out_dense_465318:	@
out_dense_465320:	
identity

identity_1

identity_2¢conv1/StatefulPartitionedCall¢.conv1/kernel/Regularizer/Square/ReadVariableOp¢conv2/StatefulPartitionedCall¢.conv2/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_3/StatefulPartitionedCall¢dense1/StatefulPartitionedCall¢ dropout1/StatefulPartitionedCall¢ dropout2/StatefulPartitionedCall¢!out_dense/StatefulPartitionedCallý
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinput_14conv2d_3_465277conv2d_3_465279*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ//*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_464773
conv1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv1_465282conv1_465284*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_464796Ä
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
GPU2*0J 8 *6
f1R/
-__inference_conv1_activity_regularizer_464719u
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
valueB:Ó
'conv1/ActivityRegularizer/strided_sliceStridedSlice(conv1/ActivityRegularizer/Shape:output:06conv1/ActivityRegularizer/strided_slice/stack:output:08conv1/ActivityRegularizer/strided_slice/stack_1:output:08conv1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv1/ActivityRegularizer/CastCast0conv1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ¥
!conv1/ActivityRegularizer/truedivRealDiv2conv1/ActivityRegularizer/PartitionedCall:output:0"conv1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: å
max_pool1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_max_pool1_layer_call_and_return_conditional_losses_464728
conv2/StatefulPartitionedCallStatefulPartitionedCall"max_pool1/PartitionedCall:output:0conv2_465296conv2_465298*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_464828Ä
)conv2/ActivityRegularizer/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *6
f1R/
-__inference_conv2_activity_regularizer_464744u
conv2/ActivityRegularizer/ShapeShape&conv2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-conv2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/conv2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/conv2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
'conv2/ActivityRegularizer/strided_sliceStridedSlice(conv2/ActivityRegularizer/Shape:output:06conv2/ActivityRegularizer/strided_slice/stack:output:08conv2/ActivityRegularizer/strided_slice/stack_1:output:08conv2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2/ActivityRegularizer/CastCast0conv2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ¥
!conv2/ActivityRegularizer/truedivRealDiv2conv2/ActivityRegularizer/PartitionedCall:output:0"conv2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: æ
max_pool2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_max_pool2_layer_call_and_return_conditional_losses_464753Ù
flatten1/PartitionedCallPartitionedCall"max_pool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten1_layer_call_and_return_conditional_losses_464849è
 dropout1/StatefulPartitionedCallStatefulPartitionedCall!flatten1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout1_layer_call_and_return_conditional_losses_465002
dense1/StatefulPartitionedCallStatefulPartitionedCall)dropout1/StatefulPartitionedCall:output:0dense1_465312dense1_465314*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_464869
 dropout2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0!^dropout1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout2_layer_call_and_return_conditional_losses_464969
!out_dense/StatefulPartitionedCallStatefulPartitionedCall)dropout2/StatefulPartitionedCall:output:0out_dense_465318out_dense_465320*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_out_dense_layer_call_and_return_conditional_losses_464893
.conv1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1_465282*&
_output_shapes
:*
dtype0
conv1/kernel/Regularizer/SquareSquare6conv1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:w
conv1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv1/kernel/Regularizer/SumSum#conv1/kernel/Regularizer/Square:y:0'conv1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
conv1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
conv1/kernel/Regularizer/mulMul'conv1/kernel/Regularizer/mul/x:output:0%conv1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.conv2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2_465296*'
_output_shapes
:*
dtype0
conv2/kernel/Regularizer/SquareSquare6conv2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:w
conv2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
conv2/kernel/Regularizer/SumSum#conv2/kernel/Regularizer/Square:y:0'conv2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
conv2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
conv2/kernel/Regularizer/mulMul'conv2/kernel/Regularizer/mul/x:output:0%conv2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: z
IdentityIdentity*out_dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe

Identity_1Identity%conv1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: e

Identity_2Identity%conv2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp^conv1/StatefulPartitionedCall/^conv1/kernel/Regularizer/Square/ReadVariableOp^conv2/StatefulPartitionedCall/^conv2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall^dense1/StatefulPartitionedCall!^dropout1/StatefulPartitionedCall!^dropout2/StatefulPartitionedCall"^out_dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ22: : : : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2`
.conv1/kernel/Regularizer/Square/ReadVariableOp.conv1/kernel/Regularizer/Square/ReadVariableOp2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2`
.conv2/kernel/Regularizer/Square/ReadVariableOp.conv2/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2D
 dropout1/StatefulPartitionedCall dropout1/StatefulPartitionedCall2D
 dropout2/StatefulPartitionedCall dropout2/StatefulPartitionedCall2F
!out_dense/StatefulPartitionedCall!out_dense/StatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"
_user_specified_name
input_14


c
D__inference_dropout1_layer_call_and_return_conditional_losses_465002

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
dropout/MulMulinputsdropout/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿòC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¨
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿòq
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿòk
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò[
IdentityIdentitydropout/Mul_1:z:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿò:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
 
_user_specified_nameinputs
¶
E
)__inference_flatten1_layer_call_fn_465709

inputs
identity´
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten1_layer_call_and_return_conditional_losses_464849b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó


-__inference_sequential_4_layer_call_fn_465408

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:$
	unknown_3:
	unknown_4:	
	unknown_5:
ò@
	unknown_6:@
	unknown_7:	@
	unknown_8:	
identity¢StatefulPartitionedCallÏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: : *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_465158p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ22: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
Ó


-__inference_sequential_4_layer_call_fn_465381

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:$
	unknown_3:
	unknown_4:	
	unknown_5:
ò@
	unknown_6:@
	unknown_7:	@
	unknown_8:	
identity¢StatefulPartitionedCallÏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: : *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_464914p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ22: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
ß
b
D__inference_dropout1_layer_call_and_return_conditional_losses_464856

inputs

identity_1P
IdentityIdentityinputs*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò]

Identity_1IdentityIdentity:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿò:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
 
_user_specified_nameinputs

a
E__inference_max_pool1_layer_call_and_return_conditional_losses_464728

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

a
E__inference_max_pool2_layer_call_and_return_conditional_losses_464753

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
b
D__inference_dropout2_layer_call_and_return_conditional_losses_465777

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*·
serving_default£
E
input_149
serving_default_input_14:0ÿÿÿÿÿÿÿÿÿ22>
	out_dense1
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:É
ê
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
»

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B_random_generator
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Ekernel
Fbias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q_random_generator
R__call__
*S&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Tkernel
Ubias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer

\iter

]beta_1

^beta_2
	_decaym©mªm«m¬*m­+m®Em¯Fm°Tm±Um²v³v´vµv¶*v·+v¸Ev¹FvºTv»Uv¼vhat½vhat¾vhat¿vhatÀ*vhatÁ+vhatÂEvhatÃFvhatÄTvhatÅUvhatÆ"
	optimizer
f
0
1
2
3
*4
+5
E6
F7
T8
U9"
trackable_list_wrapper
f
0
1
2
3
*4
+5
E6
F7
T8
U9"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
Ê
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2ÿ
-__inference_sequential_4_layer_call_fn_464939
-__inference_sequential_4_layer_call_fn_465381
-__inference_sequential_4_layer_call_fn_465408
-__inference_sequential_4_layer_call_fn_465210À
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
î2ë
H__inference_sequential_4_layer_call_and_return_conditional_losses_465490
H__inference_sequential_4_layer_call_and_return_conditional_losses_465586
H__inference_sequential_4_layer_call_and_return_conditional_losses_465274
H__inference_sequential_4_layer_call_and_return_conditional_losses_465338À
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
ÍBÊ
!__inference__wrapped_model_464706input_14"
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
gserving_default"
signature_map
):'2conv2d_3/kernel
:2conv2d_3/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_conv2d_3_layer_call_fn_465622¢
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
î2ë
D__inference_conv2d_3_layer_call_and_return_conditional_losses_465632¢
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
&:$2conv1/kernel
:2
conv1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
`0"
trackable_list_wrapper
Ê
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
ractivity_regularizer_fn
*#&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_conv1_layer_call_fn_465647¢
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
ï2ì
E__inference_conv1_layer_call_and_return_all_conditional_losses_465658¢
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_max_pool1_layer_call_fn_465663¢
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
ï2ì
E__inference_max_pool1_layer_call_and_return_conditional_losses_465668¢
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
':%2conv2/kernel
:2
conv2/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
'
a0"
trackable_list_wrapper
Ê
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
~activity_regularizer_fn
*1&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_conv2_layer_call_fn_465683¢
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
ï2ì
E__inference_conv2_layer_call_and_return_all_conditional_losses_465694¢
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_max_pool2_layer_call_fn_465699¢
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
ï2ì
E__inference_max_pool2_layer_call_and_return_conditional_losses_465704¢
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_flatten1_layer_call_fn_465709¢
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
î2ë
D__inference_flatten1_layer_call_and_return_conditional_losses_465715¢
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
)__inference_dropout1_layer_call_fn_465720
)__inference_dropout1_layer_call_fn_465725´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout1_layer_call_and_return_conditional_losses_465730
D__inference_dropout1_layer_call_and_return_conditional_losses_465742´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
!:
ò@2dense1/kernel
:@2dense1/bias
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_dense1_layer_call_fn_465751¢
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
B__inference_dense1_layer_call_and_return_conditional_losses_465762¢
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
)__inference_dropout2_layer_call_fn_465767
)__inference_dropout2_layer_call_fn_465772´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout2_layer_call_and_return_conditional_losses_465777
D__inference_dropout2_layer_call_and_return_conditional_losses_465789´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
#:!	@2out_dense/kernel
:2out_dense/bias
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_out_dense_layer_call_fn_465798¢
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
ï2ì
E__inference_out_dense_layer_call_and_return_conditional_losses_465809¢
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
³2°
__inference_loss_fn_0_465820
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_1_465831
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÌBÉ
$__inference_signature_wrapper_465613input_14"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
`0"
trackable_list_wrapper
 "
trackable_dict_wrapper
Þ2Û
-__inference_conv1_activity_regularizer_464719©
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢
	
ë2è
A__inference_conv1_layer_call_and_return_conditional_losses_465848¢
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
'
a0"
trackable_list_wrapper
 "
trackable_dict_wrapper
Þ2Û
-__inference_conv2_activity_regularizer_464744©
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢
	
ë2è
A__inference_conv2_layer_call_and_return_conditional_losses_465865¢
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
R

 total

¡count
¢	variables
£	keras_api"
_tf_keras_metric
c

¤total

¥count
¦
_fn_kwargs
§	variables
¨	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
 0
¡1"
trackable_list_wrapper
.
¢	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
¤0
¥1"
trackable_list_wrapper
.
§	variables"
_generic_user_object
.:,2Adam/conv2d_3/kernel/m
 :2Adam/conv2d_3/bias/m
+:)2Adam/conv1/kernel/m
:2Adam/conv1/bias/m
,:*2Adam/conv2/kernel/m
:2Adam/conv2/bias/m
&:$
ò@2Adam/dense1/kernel/m
:@2Adam/dense1/bias/m
(:&	@2Adam/out_dense/kernel/m
": 2Adam/out_dense/bias/m
.:,2Adam/conv2d_3/kernel/v
 :2Adam/conv2d_3/bias/v
+:)2Adam/conv1/kernel/v
:2Adam/conv1/bias/v
,:*2Adam/conv2/kernel/v
:2Adam/conv2/bias/v
&:$
ò@2Adam/dense1/kernel/v
:@2Adam/dense1/bias/v
(:&	@2Adam/out_dense/kernel/v
": 2Adam/out_dense/bias/v
1:/2Adam/conv2d_3/kernel/vhat
#:!2Adam/conv2d_3/bias/vhat
.:,2Adam/conv1/kernel/vhat
 :2Adam/conv1/bias/vhat
/:-2Adam/conv2/kernel/vhat
!:2Adam/conv2/bias/vhat
):'
ò@2Adam/dense1/kernel/vhat
!:@2Adam/dense1/bias/vhat
+:)	@2Adam/out_dense/kernel/vhat
%:#2Adam/out_dense/bias/vhat¤
!__inference__wrapped_model_464706
*+EFTU9¢6
/¢,
*'
input_14ÿÿÿÿÿÿÿÿÿ22
ª "6ª3
1
	out_dense$!
	out_denseÿÿÿÿÿÿÿÿÿW
-__inference_conv1_activity_regularizer_464719&¢
¢
	
x
ª " Ã
E__inference_conv1_layer_call_and_return_all_conditional_losses_465658z7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ//
ª ";¢8
# 
0ÿÿÿÿÿÿÿÿÿ..

	
1/0 ±
A__inference_conv1_layer_call_and_return_conditional_losses_465848l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ//
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ..
 
&__inference_conv1_layer_call_fn_465647_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ//
ª " ÿÿÿÿÿÿÿÿÿ..W
-__inference_conv2_activity_regularizer_464744&¢
¢
	
x
ª " Ä
E__inference_conv2_layer_call_and_return_all_conditional_losses_465694{*+7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "<¢9
$!
0ÿÿÿÿÿÿÿÿÿ

	
1/0 ²
A__inference_conv2_layer_call_and_return_conditional_losses_465865m*+7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
&__inference_conv2_layer_call_fn_465683`*+7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ´
D__inference_conv2d_3_layer_call_and_return_conditional_losses_465632l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ//
 
)__inference_conv2d_3_layer_call_fn_465622_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22
ª " ÿÿÿÿÿÿÿÿÿ//¤
B__inference_dense1_layer_call_and_return_conditional_losses_465762^EF1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿò
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 |
'__inference_dense1_layer_call_fn_465751QEF1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿò
ª "ÿÿÿÿÿÿÿÿÿ@¨
D__inference_dropout1_layer_call_and_return_conditional_losses_465730`5¢2
+¢(
"
inputsÿÿÿÿÿÿÿÿÿò
p 
ª "'¢$

0ÿÿÿÿÿÿÿÿÿò
 ¨
D__inference_dropout1_layer_call_and_return_conditional_losses_465742`5¢2
+¢(
"
inputsÿÿÿÿÿÿÿÿÿò
p
ª "'¢$

0ÿÿÿÿÿÿÿÿÿò
 
)__inference_dropout1_layer_call_fn_465720S5¢2
+¢(
"
inputsÿÿÿÿÿÿÿÿÿò
p 
ª "ÿÿÿÿÿÿÿÿÿò
)__inference_dropout1_layer_call_fn_465725S5¢2
+¢(
"
inputsÿÿÿÿÿÿÿÿÿò
p
ª "ÿÿÿÿÿÿÿÿÿò¤
D__inference_dropout2_layer_call_and_return_conditional_losses_465777\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¤
D__inference_dropout2_layer_call_and_return_conditional_losses_465789\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 |
)__inference_dropout2_layer_call_fn_465767O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@|
)__inference_dropout2_layer_call_fn_465772O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@«
D__inference_flatten1_layer_call_and_return_conditional_losses_465715c8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "'¢$

0ÿÿÿÿÿÿÿÿÿò
 
)__inference_flatten1_layer_call_fn_465709V8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿò;
__inference_loss_fn_0_465820¢

¢ 
ª " ;
__inference_loss_fn_1_465831*¢

¢ 
ª " è
E__inference_max_pool1_layer_call_and_return_conditional_losses_465668R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 À
*__inference_max_pool1_layer_call_fn_465663R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿè
E__inference_max_pool2_layer_call_and_return_conditional_losses_465704R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 À
*__inference_max_pool2_layer_call_fn_465699R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
E__inference_out_dense_layer_call_and_return_conditional_losses_465809]TU/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
*__inference_out_dense_layer_call_fn_465798PTU/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿà
H__inference_sequential_4_layer_call_and_return_conditional_losses_465274
*+EFTUA¢>
7¢4
*'
input_14ÿÿÿÿÿÿÿÿÿ22
p 

 
ª "B¢?

0ÿÿÿÿÿÿÿÿÿ

	
1/0 
	
1/1 à
H__inference_sequential_4_layer_call_and_return_conditional_losses_465338
*+EFTUA¢>
7¢4
*'
input_14ÿÿÿÿÿÿÿÿÿ22
p

 
ª "B¢?

0ÿÿÿÿÿÿÿÿÿ

	
1/0 
	
1/1 Þ
H__inference_sequential_4_layer_call_and_return_conditional_losses_465490
*+EFTU?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22
p 

 
ª "B¢?

0ÿÿÿÿÿÿÿÿÿ

	
1/0 
	
1/1 Þ
H__inference_sequential_4_layer_call_and_return_conditional_losses_465586
*+EFTU?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22
p

 
ª "B¢?

0ÿÿÿÿÿÿÿÿÿ

	
1/0 
	
1/1 
-__inference_sequential_4_layer_call_fn_464939j
*+EFTUA¢>
7¢4
*'
input_14ÿÿÿÿÿÿÿÿÿ22
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_4_layer_call_fn_465210j
*+EFTUA¢>
7¢4
*'
input_14ÿÿÿÿÿÿÿÿÿ22
p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_4_layer_call_fn_465381h
*+EFTU?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_4_layer_call_fn_465408h
*+EFTU?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22
p

 
ª "ÿÿÿÿÿÿÿÿÿ´
$__inference_signature_wrapper_465613
*+EFTUE¢B
¢ 
;ª8
6
input_14*'
input_14ÿÿÿÿÿÿÿÿÿ22"6ª3
1
	out_dense$!
	out_denseÿÿÿÿÿÿÿÿÿ