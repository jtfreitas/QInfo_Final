??4
?$?$
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
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
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
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??3
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
?
conv2dmpo/end_node_firstVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2dmpo/end_node_first
?
,conv2dmpo/end_node_first/Read/ReadVariableOpReadVariableOpconv2dmpo/end_node_first*&
_output_shapes
:*
dtype0
?
conv2dmpo/end_node_lastVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2dmpo/end_node_last
?
+conv2dmpo/end_node_last/Read/ReadVariableOpReadVariableOpconv2dmpo/end_node_last*&
_output_shapes
:*
dtype0
t
conv2dmpo/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2dmpo/bias
m
"conv2dmpo/bias/Read/ReadVariableOpReadVariableOpconv2dmpo/bias*
_output_shapes
:*
dtype0
?
conv2dmpo_1/end_node_firstVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv2dmpo_1/end_node_first
?
.conv2dmpo_1/end_node_first/Read/ReadVariableOpReadVariableOpconv2dmpo_1/end_node_first*&
_output_shapes
:*
dtype0
?
conv2dmpo_1/middle_node_0VarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2dmpo_1/middle_node_0
?
-conv2dmpo_1/middle_node_0/Read/ReadVariableOpReadVariableOpconv2dmpo_1/middle_node_0*&
_output_shapes
:*
dtype0
?
conv2dmpo_1/middle_node_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2dmpo_1/middle_node_1
?
-conv2dmpo_1/middle_node_1/Read/ReadVariableOpReadVariableOpconv2dmpo_1/middle_node_1*&
_output_shapes
:*
dtype0
?
conv2dmpo_1/end_node_lastVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2dmpo_1/end_node_last
?
-conv2dmpo_1/end_node_last/Read/ReadVariableOpReadVariableOpconv2dmpo_1/end_node_last*&
_output_shapes
:*
dtype0
y
conv2dmpo_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameconv2dmpo_1/bias
r
$conv2dmpo_1/bias/Read/ReadVariableOpReadVariableOpconv2dmpo_1/bias*
_output_shapes	
:?*
dtype0
b
aVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namea
[
a/Read/ReadVariableOpReadVariableOpa*"
_output_shapes
:*
dtype0
g
bVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameb
`
b/Read/ReadVariableOpReadVariableOpb*'
_output_shapes
:?*
dtype0
b
cVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namec
[
c/Read/ReadVariableOpReadVariableOpc*"
_output_shapes
:*
dtype0
h
biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namebias
a
bias/Read/ReadVariableOpReadVariableOpbias*"
_output_shapes
:*
dtype0
b
a_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namea_1
[
a_1/Read/ReadVariableOpReadVariableOpa_1*
_output_shapes

:*
dtype0
k
b_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameb_1
d
b_1/Read/ReadVariableOpReadVariableOpb_1*'
_output_shapes
:?*
dtype0
b
c_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namec_1
[
c_1/Read/ReadVariableOpReadVariableOpc_1*
_output_shapes

:*
dtype0
e
bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namebias_1
^
bias_1/Read/ReadVariableOpReadVariableOpbias_1*
_output_shapes	
:?*
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
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2dmpo/end_node_first/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2dmpo/end_node_first/m
?
3Adam/conv2dmpo/end_node_first/m/Read/ReadVariableOpReadVariableOpAdam/conv2dmpo/end_node_first/m*&
_output_shapes
:*
dtype0
?
Adam/conv2dmpo/end_node_last/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2dmpo/end_node_last/m
?
2Adam/conv2dmpo/end_node_last/m/Read/ReadVariableOpReadVariableOpAdam/conv2dmpo/end_node_last/m*&
_output_shapes
:*
dtype0
?
Adam/conv2dmpo/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2dmpo/bias/m
{
)Adam/conv2dmpo/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2dmpo/bias/m*
_output_shapes
:*
dtype0
?
!Adam/conv2dmpo_1/end_node_first/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2dmpo_1/end_node_first/m
?
5Adam/conv2dmpo_1/end_node_first/m/Read/ReadVariableOpReadVariableOp!Adam/conv2dmpo_1/end_node_first/m*&
_output_shapes
:*
dtype0
?
 Adam/conv2dmpo_1/middle_node_0/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2dmpo_1/middle_node_0/m
?
4Adam/conv2dmpo_1/middle_node_0/m/Read/ReadVariableOpReadVariableOp Adam/conv2dmpo_1/middle_node_0/m*&
_output_shapes
:*
dtype0
?
 Adam/conv2dmpo_1/middle_node_1/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2dmpo_1/middle_node_1/m
?
4Adam/conv2dmpo_1/middle_node_1/m/Read/ReadVariableOpReadVariableOp Adam/conv2dmpo_1/middle_node_1/m*&
_output_shapes
:*
dtype0
?
 Adam/conv2dmpo_1/end_node_last/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2dmpo_1/end_node_last/m
?
4Adam/conv2dmpo_1/end_node_last/m/Read/ReadVariableOpReadVariableOp Adam/conv2dmpo_1/end_node_last/m*&
_output_shapes
:*
dtype0
?
Adam/conv2dmpo_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/conv2dmpo_1/bias/m
?
+Adam/conv2dmpo_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2dmpo_1/bias/m*
_output_shapes	
:?*
dtype0
p
Adam/a/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Adam/a/m
i
Adam/a/m/Read/ReadVariableOpReadVariableOpAdam/a/m*"
_output_shapes
:*
dtype0
u
Adam/b/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
Adam/b/m
n
Adam/b/m/Read/ReadVariableOpReadVariableOpAdam/b/m*'
_output_shapes
:?*
dtype0
p
Adam/c/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Adam/c/m
i
Adam/c/m/Read/ReadVariableOpReadVariableOpAdam/c/m*"
_output_shapes
:*
dtype0
v
Adam/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/bias/m
o
Adam/bias/m/Read/ReadVariableOpReadVariableOpAdam/bias/m*"
_output_shapes
:*
dtype0
p

Adam/a/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
Adam/a/m_1
i
Adam/a/m_1/Read/ReadVariableOpReadVariableOp
Adam/a/m_1*
_output_shapes

:*
dtype0
y

Adam/b/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
Adam/b/m_1
r
Adam/b/m_1/Read/ReadVariableOpReadVariableOp
Adam/b/m_1*'
_output_shapes
:?*
dtype0
p

Adam/c/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
Adam/c/m_1
i
Adam/c/m_1/Read/ReadVariableOpReadVariableOp
Adam/c/m_1*
_output_shapes

:*
dtype0
s
Adam/bias/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameAdam/bias/m_1
l
!Adam/bias/m_1/Read/ReadVariableOpReadVariableOpAdam/bias/m_1*
_output_shapes	
:?*
dtype0
?
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2dmpo/end_node_first/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2dmpo/end_node_first/v
?
3Adam/conv2dmpo/end_node_first/v/Read/ReadVariableOpReadVariableOpAdam/conv2dmpo/end_node_first/v*&
_output_shapes
:*
dtype0
?
Adam/conv2dmpo/end_node_last/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2dmpo/end_node_last/v
?
2Adam/conv2dmpo/end_node_last/v/Read/ReadVariableOpReadVariableOpAdam/conv2dmpo/end_node_last/v*&
_output_shapes
:*
dtype0
?
Adam/conv2dmpo/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2dmpo/bias/v
{
)Adam/conv2dmpo/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2dmpo/bias/v*
_output_shapes
:*
dtype0
?
!Adam/conv2dmpo_1/end_node_first/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2dmpo_1/end_node_first/v
?
5Adam/conv2dmpo_1/end_node_first/v/Read/ReadVariableOpReadVariableOp!Adam/conv2dmpo_1/end_node_first/v*&
_output_shapes
:*
dtype0
?
 Adam/conv2dmpo_1/middle_node_0/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2dmpo_1/middle_node_0/v
?
4Adam/conv2dmpo_1/middle_node_0/v/Read/ReadVariableOpReadVariableOp Adam/conv2dmpo_1/middle_node_0/v*&
_output_shapes
:*
dtype0
?
 Adam/conv2dmpo_1/middle_node_1/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2dmpo_1/middle_node_1/v
?
4Adam/conv2dmpo_1/middle_node_1/v/Read/ReadVariableOpReadVariableOp Adam/conv2dmpo_1/middle_node_1/v*&
_output_shapes
:*
dtype0
?
 Adam/conv2dmpo_1/end_node_last/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2dmpo_1/end_node_last/v
?
4Adam/conv2dmpo_1/end_node_last/v/Read/ReadVariableOpReadVariableOp Adam/conv2dmpo_1/end_node_last/v*&
_output_shapes
:*
dtype0
?
Adam/conv2dmpo_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/conv2dmpo_1/bias/v
?
+Adam/conv2dmpo_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2dmpo_1/bias/v*
_output_shapes	
:?*
dtype0
p
Adam/a/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Adam/a/v
i
Adam/a/v/Read/ReadVariableOpReadVariableOpAdam/a/v*"
_output_shapes
:*
dtype0
u
Adam/b/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
Adam/b/v
n
Adam/b/v/Read/ReadVariableOpReadVariableOpAdam/b/v*'
_output_shapes
:?*
dtype0
p
Adam/c/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Adam/c/v
i
Adam/c/v/Read/ReadVariableOpReadVariableOpAdam/c/v*"
_output_shapes
:*
dtype0
v
Adam/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/bias/v
o
Adam/bias/v/Read/ReadVariableOpReadVariableOpAdam/bias/v*"
_output_shapes
:*
dtype0
p

Adam/a/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
Adam/a/v_1
i
Adam/a/v_1/Read/ReadVariableOpReadVariableOp
Adam/a/v_1*
_output_shapes

:*
dtype0
y

Adam/b/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
Adam/b/v_1
r
Adam/b/v_1/Read/ReadVariableOpReadVariableOp
Adam/b/v_1*'
_output_shapes
:?*
dtype0
p

Adam/c/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
Adam/c/v_1
i
Adam/c/v_1/Read/ReadVariableOpReadVariableOp
Adam/c/v_1*
_output_shapes

:*
dtype0
s
Adam/bias/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameAdam/bias/v_1
l
!Adam/bias/v_1/Read/ReadVariableOpReadVariableOpAdam/bias/v_1*
_output_shapes	
:?*
dtype0
?
Adam/conv2d/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d/kernel/vhat
?
+Adam/conv2d/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/vhat*&
_output_shapes
:*
dtype0
?
Adam/conv2d/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d/bias/vhat
{
)Adam/conv2d/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/vhat*
_output_shapes
:*
dtype0
?
"Adam/conv2dmpo/end_node_first/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/conv2dmpo/end_node_first/vhat
?
6Adam/conv2dmpo/end_node_first/vhat/Read/ReadVariableOpReadVariableOp"Adam/conv2dmpo/end_node_first/vhat*&
_output_shapes
:*
dtype0
?
!Adam/conv2dmpo/end_node_last/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2dmpo/end_node_last/vhat
?
5Adam/conv2dmpo/end_node_last/vhat/Read/ReadVariableOpReadVariableOp!Adam/conv2dmpo/end_node_last/vhat*&
_output_shapes
:*
dtype0
?
Adam/conv2dmpo/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2dmpo/bias/vhat
?
,Adam/conv2dmpo/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/conv2dmpo/bias/vhat*
_output_shapes
:*
dtype0
?
$Adam/conv2dmpo_1/end_node_first/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/conv2dmpo_1/end_node_first/vhat
?
8Adam/conv2dmpo_1/end_node_first/vhat/Read/ReadVariableOpReadVariableOp$Adam/conv2dmpo_1/end_node_first/vhat*&
_output_shapes
:*
dtype0
?
#Adam/conv2dmpo_1/middle_node_0/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/conv2dmpo_1/middle_node_0/vhat
?
7Adam/conv2dmpo_1/middle_node_0/vhat/Read/ReadVariableOpReadVariableOp#Adam/conv2dmpo_1/middle_node_0/vhat*&
_output_shapes
:*
dtype0
?
#Adam/conv2dmpo_1/middle_node_1/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/conv2dmpo_1/middle_node_1/vhat
?
7Adam/conv2dmpo_1/middle_node_1/vhat/Read/ReadVariableOpReadVariableOp#Adam/conv2dmpo_1/middle_node_1/vhat*&
_output_shapes
:*
dtype0
?
#Adam/conv2dmpo_1/end_node_last/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/conv2dmpo_1/end_node_last/vhat
?
7Adam/conv2dmpo_1/end_node_last/vhat/Read/ReadVariableOpReadVariableOp#Adam/conv2dmpo_1/end_node_last/vhat*&
_output_shapes
:*
dtype0
?
Adam/conv2dmpo_1/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/conv2dmpo_1/bias/vhat
?
.Adam/conv2dmpo_1/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/conv2dmpo_1/bias/vhat*
_output_shapes	
:?*
dtype0
v
Adam/a/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/a/vhat
o
Adam/a/vhat/Read/ReadVariableOpReadVariableOpAdam/a/vhat*"
_output_shapes
:*
dtype0
{
Adam/b/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameAdam/b/vhat
t
Adam/b/vhat/Read/ReadVariableOpReadVariableOpAdam/b/vhat*'
_output_shapes
:?*
dtype0
v
Adam/c/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/c/vhat
o
Adam/c/vhat/Read/ReadVariableOpReadVariableOpAdam/c/vhat*"
_output_shapes
:*
dtype0
|
Adam/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/bias/vhat
u
"Adam/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/bias/vhat*"
_output_shapes
:*
dtype0
v
Adam/a/vhat_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameAdam/a/vhat_1
o
!Adam/a/vhat_1/Read/ReadVariableOpReadVariableOpAdam/a/vhat_1*
_output_shapes

:*
dtype0

Adam/b/vhat_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameAdam/b/vhat_1
x
!Adam/b/vhat_1/Read/ReadVariableOpReadVariableOpAdam/b/vhat_1*'
_output_shapes
:?*
dtype0
v
Adam/c/vhat_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameAdam/c/vhat_1
o
!Adam/c/vhat_1/Read/ReadVariableOpReadVariableOpAdam/c/vhat_1*
_output_shapes

:*
dtype0
y
Adam/bias/vhat_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/bias/vhat_1
r
$Adam/bias/vhat_1/Read/ReadVariableOpReadVariableOpAdam/bias/vhat_1*
_output_shapes	
:?*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?~B?~ B?~
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?
	nodes
end_node_first
end_node_last
bias
bias_var
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses*
?
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses* 
?
	*nodes
+end_node_first
,middle_node_0
-middle_node_1
.end_node_last
/bias
/bias_var
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses*
?
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses* 
?
	<a_var
	=b_var
	>c_var
?bias
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses*
?
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J_random_generator
K__call__
*L&call_and_return_all_conditional_losses* 
?
	Ma_var
	Nb_var
	Oc_var
Pbias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses*
?
Witer

Xbeta_1

Ybeta_2
	Zdecaym?m?m?m?m?+m?,m?-m?.m?/m?<m?=m?>m??m?Mm?Nm?Om?Pm?v?v?v?v?v?+v?,v?-v?.v?/v?<v?=v?>v??v?Mv?Nv?Ov?Pv?vhat?vhat?vhat?vhat?vhat?+vhat?,vhat?-vhat?.vhat?/vhat?<vhat?=vhat?>vhat??vhat?Mvhat?Nvhat?Ovhat?Pvhat?*
?
0
1
2
3
4
+5
,6
-7
.8
/9
<10
=11
>12
?13
M14
N15
O16
P17*
?
0
1
2
3
4
+5
,6
-7
.8
/9
<10
=11
>12
?13
M14
N15
O16
P17*
* 
?
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

`serving_default* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
?
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 

0
1*
pj
VARIABLE_VALUEconv2dmpo/end_node_first>layer_with_weights-1/end_node_first/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEconv2dmpo/end_node_last=layer_with_weights-1/end_node_last/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2dmpo/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1
2*

0
1
2*
* 
?
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
kactivity_regularizer_fn
*#&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses* 
* 
* 
 
+0
,1
-2
.3*
rl
VARIABLE_VALUEconv2dmpo_1/end_node_first>layer_with_weights-2/end_node_first/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEconv2dmpo_1/middle_node_0=layer_with_weights-2/middle_node_0/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEconv2dmpo_1/middle_node_1=layer_with_weights-2/middle_node_1/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEconv2dmpo_1/end_node_last=layer_with_weights-2/end_node_last/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEconv2dmpo_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
'
+0
,1
-2
.3
/4*
'
+0
,1
-2
.3
/4*
* 
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
wactivity_regularizer_fn
*5&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses* 
* 
* 
PJ
VARIABLE_VALUEa5layer_with_weights-3/a_var/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEb5layer_with_weights-3/b_var/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEc5layer_with_weights-3/c_var/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEbias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
<0
=1
>2
?3*
 
<0
=1
>2
?3*
* 
?
~non_trainable_variables

layers
?metrics
 ?layer_regularization_losses
?layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
?activity_regularizer_fn
*E&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses* 
* 
* 
* 
RL
VARIABLE_VALUEa_15layer_with_weights-4/a_var/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEb_15layer_with_weights-4/b_var/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEc_15layer_with_weights-4/c_var/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbias_14layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
M0
N1
O2
P3*
 
M0
N1
O2
P3*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
?activity_regularizer_fn
*V&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
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
<
0
1
2
3
4
5
6
7*

?0
?1*
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

?total

?count
?	variables
?	keras_api*
`
?
thresholds
?true_positives
?false_positives
?	variables
?	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
* 
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
?z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2dmpo/end_node_first/mZlayer_with_weights-1/end_node_first/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2dmpo/end_node_last/mYlayer_with_weights-1/end_node_last/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2dmpo/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2dmpo_1/end_node_first/mZlayer_with_weights-2/end_node_first/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/conv2dmpo_1/middle_node_0/mYlayer_with_weights-2/middle_node_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/conv2dmpo_1/middle_node_1/mYlayer_with_weights-2/middle_node_1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/conv2dmpo_1/end_node_last/mYlayer_with_weights-2/end_node_last/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/conv2dmpo_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/a/mQlayer_with_weights-3/a_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/b/mQlayer_with_weights-3/b_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/c/mQlayer_with_weights-3/c_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE
Adam/a/m_1Qlayer_with_weights-4/a_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE
Adam/b/m_1Qlayer_with_weights-4/b_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE
Adam/c/m_1Qlayer_with_weights-4/c_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/bias/m_1Player_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2dmpo/end_node_first/vZlayer_with_weights-1/end_node_first/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2dmpo/end_node_last/vYlayer_with_weights-1/end_node_last/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2dmpo/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2dmpo_1/end_node_first/vZlayer_with_weights-2/end_node_first/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/conv2dmpo_1/middle_node_0/vYlayer_with_weights-2/middle_node_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/conv2dmpo_1/middle_node_1/vYlayer_with_weights-2/middle_node_1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/conv2dmpo_1/end_node_last/vYlayer_with_weights-2/end_node_last/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/conv2dmpo_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/a/vQlayer_with_weights-3/a_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/b/vQlayer_with_weights-3/b_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/c/vQlayer_with_weights-3/c_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE
Adam/a/v_1Qlayer_with_weights-4/a_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE
Adam/b/v_1Qlayer_with_weights-4/b_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE
Adam/c/v_1Qlayer_with_weights-4/c_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/bias/v_1Player_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d/kernel/vhatUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/conv2d/bias/vhatSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/conv2dmpo/end_node_first/vhat]layer_with_weights-1/end_node_first/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2dmpo/end_node_last/vhat\layer_with_weights-1/end_node_last/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEAdam/conv2dmpo/bias/vhatSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam/conv2dmpo_1/end_node_first/vhat]layer_with_weights-2/end_node_first/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/conv2dmpo_1/middle_node_0/vhat\layer_with_weights-2/middle_node_0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/conv2dmpo_1/middle_node_1/vhat\layer_with_weights-2/middle_node_1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/conv2dmpo_1/end_node_last/vhat\layer_with_weights-2/end_node_last/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2dmpo_1/bias/vhatSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/a/vhatTlayer_with_weights-3/a_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/b/vhatTlayer_with_weights-3/b_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/c/vhatTlayer_with_weights-3/c_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/bias/vhatSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/a/vhat_1Tlayer_with_weights-4/a_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/b/vhat_1Tlayer_with_weights-4/b_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/c/vhat_1Tlayer_with_weights-4/c_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/bias/vhat_1Slayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_input_12Placeholder*/
_output_shapes
:?????????22*
dtype0*$
shape:?????????22
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_12conv2d/kernelconv2d/biasconv2dmpo/end_node_firstconv2dmpo/end_node_lastconv2dmpo/biasconv2dmpo_1/end_node_firstconv2dmpo_1/middle_node_0conv2dmpo_1/middle_node_1conv2dmpo_1/end_node_lastconv2dmpo_1/biasabcbiasa_1b_1c_1bias_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_415540
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp,conv2dmpo/end_node_first/Read/ReadVariableOp+conv2dmpo/end_node_last/Read/ReadVariableOp"conv2dmpo/bias/Read/ReadVariableOp.conv2dmpo_1/end_node_first/Read/ReadVariableOp-conv2dmpo_1/middle_node_0/Read/ReadVariableOp-conv2dmpo_1/middle_node_1/Read/ReadVariableOp-conv2dmpo_1/end_node_last/Read/ReadVariableOp$conv2dmpo_1/bias/Read/ReadVariableOpa/Read/ReadVariableOpb/Read/ReadVariableOpc/Read/ReadVariableOpbias/Read/ReadVariableOpa_1/Read/ReadVariableOpb_1/Read/ReadVariableOpc_1/Read/ReadVariableOpbias_1/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp3Adam/conv2dmpo/end_node_first/m/Read/ReadVariableOp2Adam/conv2dmpo/end_node_last/m/Read/ReadVariableOp)Adam/conv2dmpo/bias/m/Read/ReadVariableOp5Adam/conv2dmpo_1/end_node_first/m/Read/ReadVariableOp4Adam/conv2dmpo_1/middle_node_0/m/Read/ReadVariableOp4Adam/conv2dmpo_1/middle_node_1/m/Read/ReadVariableOp4Adam/conv2dmpo_1/end_node_last/m/Read/ReadVariableOp+Adam/conv2dmpo_1/bias/m/Read/ReadVariableOpAdam/a/m/Read/ReadVariableOpAdam/b/m/Read/ReadVariableOpAdam/c/m/Read/ReadVariableOpAdam/bias/m/Read/ReadVariableOpAdam/a/m_1/Read/ReadVariableOpAdam/b/m_1/Read/ReadVariableOpAdam/c/m_1/Read/ReadVariableOp!Adam/bias/m_1/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp3Adam/conv2dmpo/end_node_first/v/Read/ReadVariableOp2Adam/conv2dmpo/end_node_last/v/Read/ReadVariableOp)Adam/conv2dmpo/bias/v/Read/ReadVariableOp5Adam/conv2dmpo_1/end_node_first/v/Read/ReadVariableOp4Adam/conv2dmpo_1/middle_node_0/v/Read/ReadVariableOp4Adam/conv2dmpo_1/middle_node_1/v/Read/ReadVariableOp4Adam/conv2dmpo_1/end_node_last/v/Read/ReadVariableOp+Adam/conv2dmpo_1/bias/v/Read/ReadVariableOpAdam/a/v/Read/ReadVariableOpAdam/b/v/Read/ReadVariableOpAdam/c/v/Read/ReadVariableOpAdam/bias/v/Read/ReadVariableOpAdam/a/v_1/Read/ReadVariableOpAdam/b/v_1/Read/ReadVariableOpAdam/c/v_1/Read/ReadVariableOp!Adam/bias/v_1/Read/ReadVariableOp+Adam/conv2d/kernel/vhat/Read/ReadVariableOp)Adam/conv2d/bias/vhat/Read/ReadVariableOp6Adam/conv2dmpo/end_node_first/vhat/Read/ReadVariableOp5Adam/conv2dmpo/end_node_last/vhat/Read/ReadVariableOp,Adam/conv2dmpo/bias/vhat/Read/ReadVariableOp8Adam/conv2dmpo_1/end_node_first/vhat/Read/ReadVariableOp7Adam/conv2dmpo_1/middle_node_0/vhat/Read/ReadVariableOp7Adam/conv2dmpo_1/middle_node_1/vhat/Read/ReadVariableOp7Adam/conv2dmpo_1/end_node_last/vhat/Read/ReadVariableOp.Adam/conv2dmpo_1/bias/vhat/Read/ReadVariableOpAdam/a/vhat/Read/ReadVariableOpAdam/b/vhat/Read/ReadVariableOpAdam/c/vhat/Read/ReadVariableOp"Adam/bias/vhat/Read/ReadVariableOp!Adam/a/vhat_1/Read/ReadVariableOp!Adam/b/vhat_1/Read/ReadVariableOp!Adam/c/vhat_1/Read/ReadVariableOp$Adam/bias/vhat_1/Read/ReadVariableOpConst*]
TinV
T2R	*
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
__inference__traced_save_416732
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2dmpo/end_node_firstconv2dmpo/end_node_lastconv2dmpo/biasconv2dmpo_1/end_node_firstconv2dmpo_1/middle_node_0conv2dmpo_1/middle_node_1conv2dmpo_1/end_node_lastconv2dmpo_1/biasabcbiasa_1b_1c_1bias_1	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcounttrue_positivesfalse_positivesAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2dmpo/end_node_first/mAdam/conv2dmpo/end_node_last/mAdam/conv2dmpo/bias/m!Adam/conv2dmpo_1/end_node_first/m Adam/conv2dmpo_1/middle_node_0/m Adam/conv2dmpo_1/middle_node_1/m Adam/conv2dmpo_1/end_node_last/mAdam/conv2dmpo_1/bias/mAdam/a/mAdam/b/mAdam/c/mAdam/bias/m
Adam/a/m_1
Adam/b/m_1
Adam/c/m_1Adam/bias/m_1Adam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2dmpo/end_node_first/vAdam/conv2dmpo/end_node_last/vAdam/conv2dmpo/bias/v!Adam/conv2dmpo_1/end_node_first/v Adam/conv2dmpo_1/middle_node_0/v Adam/conv2dmpo_1/middle_node_1/v Adam/conv2dmpo_1/end_node_last/vAdam/conv2dmpo_1/bias/vAdam/a/vAdam/b/vAdam/c/vAdam/bias/v
Adam/a/v_1
Adam/b/v_1
Adam/c/v_1Adam/bias/v_1Adam/conv2d/kernel/vhatAdam/conv2d/bias/vhat"Adam/conv2dmpo/end_node_first/vhat!Adam/conv2dmpo/end_node_last/vhatAdam/conv2dmpo/bias/vhat$Adam/conv2dmpo_1/end_node_first/vhat#Adam/conv2dmpo_1/middle_node_0/vhat#Adam/conv2dmpo_1/middle_node_1/vhat#Adam/conv2dmpo_1/end_node_last/vhatAdam/conv2dmpo_1/bias/vhatAdam/a/vhatAdam/b/vhatAdam/c/vhatAdam/bias/vhatAdam/a/vhat_1Adam/b/vhat_1Adam/c/vhat_1Adam/bias/vhat_1*\
TinU
S2Q*
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
"__inference__traced_restore_416982??0
??
?
E__inference_fruit_tn2_layer_call_and_return_conditional_losses_416469

inputs3
!loop_body_readvariableop_resource:>
#loop_body_readvariableop_1_resource:?5
#loop_body_readvariableop_2_resource:4
%loop_body_add_readvariableop_resource:	?
identity??loop_body/ReadVariableOp?loop_body/ReadVariableOp_1?loop_body/ReadVariableOp_2?loop_body/add/ReadVariableOp;
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
Tparams0*
_output_shapes
:@l
loop_body/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
loop_body/ReshapeReshapeloop_body/GatherV2:output:0 loop_body/Reshape/shape:output:0*
T0*"
_output_shapes
:z
loop_body/ReadVariableOpReadVariableOp!loop_body_readvariableop_resource*
_output_shapes

:*
dtype0?
loop_body/ReadVariableOp_1ReadVariableOp#loop_body_readvariableop_1_resource*'
_output_shapes
:?*
dtype0~
loop_body/ReadVariableOp_2ReadVariableOp#loop_body_readvariableop_2_resource*
_output_shapes

:*
dtype0w
"loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
loop_body/Tensordot/transpose	Transposeloop_body/Reshape:output:0+loop_body/Tensordot/transpose/perm:output:0*
T0*"
_output_shapes
:r
!loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
loop_body/Tensordot/ReshapeReshape!loop_body/Tensordot/transpose:y:0*loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:?
loop_body/Tensordot/MatMulMatMul$loop_body/Tensordot/Reshape:output:0 loop_body/ReadVariableOp:value:0*
T0*
_output_shapes

:n
loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
loop_body/TensordotReshape$loop_body/Tensordot/MatMul:product:0"loop_body/Tensordot/shape:output:0*
T0*"
_output_shapes
:}
$loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
loop_body/Tensordot_1/transpose	Transpose"loop_body/ReadVariableOp_1:value:0-loop_body/Tensordot_1/transpose/perm:output:0*
T0*'
_output_shapes
:?t
#loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"     ?
loop_body/Tensordot_1/ReshapeReshape#loop_body/Tensordot_1/transpose:y:0,loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes
:	?{
&loop_body/Tensordot_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
!loop_body/Tensordot_1/transpose_1	Transposeloop_body/Tensordot:output:0/loop_body/Tensordot_1/transpose_1/perm:output:0*
T0*"
_output_shapes
:v
%loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
loop_body/Tensordot_1/Reshape_1Reshape%loop_body/Tensordot_1/transpose_1:y:0.loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:?
loop_body/Tensordot_1/MatMulMatMul&loop_body/Tensordot_1/Reshape:output:0(loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	?p
loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"?         ?
loop_body/Tensordot_1Reshape&loop_body/Tensordot_1/MatMul:product:0$loop_body/Tensordot_1/shape:output:0*
T0*#
_output_shapes
:?t
#loop_body/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
loop_body/Tensordot_2/ReshapeReshape"loop_body/ReadVariableOp_2:value:0,loop_body/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes

:y
$loop_body/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
loop_body/Tensordot_2/transpose	Transposeloop_body/Tensordot_1:output:0-loop_body/Tensordot_2/transpose/perm:output:0*
T0*#
_output_shapes
:?v
%loop_body/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   ?
loop_body/Tensordot_2/Reshape_1Reshape#loop_body/Tensordot_2/transpose:y:0.loop_body/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	??
loop_body/Tensordot_2/MatMulMatMul&loop_body/Tensordot_2/Reshape:output:0(loop_body/Tensordot_2/Reshape_1:output:0*
T0*
_output_shapes
:	?f
loop_body/Tensordot_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:??
loop_body/Tensordot_2Reshape&loop_body/Tensordot_2/MatMul:product:0$loop_body/Tensordot_2/shape:output:0*
T0*
_output_shapes	
:?
loop_body/add/ReadVariableOpReadVariableOp%loop_body_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
loop_body/addAddV2loop_body/Tensordot_2:output:0$loop_body/add/ReadVariableOp:value:0*
T0*
_output_shapes	
:?\
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
Tparams0*'
_output_shapes
:?????????@d
"loop_body/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
loop_body/Reshape/pfor/concatConcatV2pfor/Reshape:output:0 loop_body/Reshape/shape:output:0+loop_body/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
loop_body/Reshape/pfor/ReshapeReshape)loop_body/GatherV2/pfor/GatherV2:output:0&loop_body/Reshape/pfor/concat:output:0*
T0*/
_output_shapes
:?????????j
(loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
&loop_body/Tensordot/transpose/pfor/addAddV2+loop_body/Tensordot/transpose/perm:output:01loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:|
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
:?
,loop_body/Tensordot/transpose/pfor/Transpose	Transpose'loop_body/Reshape/pfor/Reshape:output:02loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:?????????n
,loop_body/Tensordot/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'loop_body/Tensordot/Reshape/pfor/concatConcatV2pfor/Reshape:output:0*loop_body/Tensordot/Reshape/shape:output:05loop_body/Tensordot/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
(loop_body/Tensordot/Reshape/pfor/ReshapeReshape0loop_body/Tensordot/transpose/pfor/Transpose:y:00loop_body/Tensordot/Reshape/pfor/concat:output:0*
T0*+
_output_shapes
:??????????
%loop_body/Tensordot/MatMul/pfor/ShapeShape1loop_body/Tensordot/Reshape/pfor/Reshape:output:0*
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
)loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape1loop_body/Tensordot/Reshape/pfor/Reshape:output:08loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:???????????????????
&loop_body/Tensordot/MatMul/pfor/MatMulMatMul2loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0 loop_body/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
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
T0*+
_output_shapes
:?????????f
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
:?????????n
,loop_body/Tensordot_1/transpose_1/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
*loop_body/Tensordot_1/transpose_1/pfor/addAddV2/loop_body/Tensordot_1/transpose_1/perm:output:05loop_body/Tensordot_1/transpose_1/pfor/add/y:output:0*
T0*
_output_shapes
:?
6loop_body/Tensordot_1/transpose_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: t
2loop_body/Tensordot_1/transpose_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-loop_body/Tensordot_1/transpose_1/pfor/concatConcatV2?loop_body/Tensordot_1/transpose_1/pfor/concat/values_0:output:0.loop_body/Tensordot_1/transpose_1/pfor/add:z:0;loop_body/Tensordot_1/transpose_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
0loop_body/Tensordot_1/transpose_1/pfor/Transpose	Transpose)loop_body/Tensordot/pfor/Reshape:output:06loop_body/Tensordot_1/transpose_1/pfor/concat:output:0*
T0*/
_output_shapes
:?????????r
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
,loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape4loop_body/Tensordot_1/transpose_1/pfor/Transpose:y:04loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*+
_output_shapes
:??????????
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
T0*+
_output_shapes
:??????????
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
T0*(
_output_shapes
:??????????~
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
T0*,
_output_shapes
:??????????h
&loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
!loop_body/Tensordot_1/pfor/concatConcatV2pfor/Reshape:output:0$loop_body/Tensordot_1/shape:output:0/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
"loop_body/Tensordot_1/pfor/ReshapeReshape1loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:0*loop_body/Tensordot_1/pfor/concat:output:0*
T0*0
_output_shapes
:??????????l
*loop_body/Tensordot_2/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
(loop_body/Tensordot_2/transpose/pfor/addAddV2-loop_body/Tensordot_2/transpose/perm:output:03loop_body/Tensordot_2/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:~
4loop_body/Tensordot_2/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: r
0loop_body/Tensordot_2/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+loop_body/Tensordot_2/transpose/pfor/concatConcatV2=loop_body/Tensordot_2/transpose/pfor/concat/values_0:output:0,loop_body/Tensordot_2/transpose/pfor/add:z:09loop_body/Tensordot_2/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
.loop_body/Tensordot_2/transpose/pfor/Transpose	Transpose+loop_body/Tensordot_1/pfor/Reshape:output:04loop_body/Tensordot_2/transpose/pfor/concat:output:0*
T0*0
_output_shapes
:??????????r
0loop_body/Tensordot_2/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+loop_body/Tensordot_2/Reshape_1/pfor/concatConcatV2pfor/Reshape:output:0.loop_body/Tensordot_2/Reshape_1/shape:output:09loop_body/Tensordot_2/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,loop_body/Tensordot_2/Reshape_1/pfor/ReshapeReshape2loop_body/Tensordot_2/transpose/pfor/Transpose:y:04loop_body/Tensordot_2/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:???????????
'loop_body/Tensordot_2/MatMul/pfor/ShapeShape5loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
5loop_body/Tensordot_2/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/loop_body/Tensordot_2/MatMul/pfor/strided_sliceStridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0>loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_1StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
)loop_body/Tensordot_2/MatMul/pfor/MinimumMinimum8loop_body/Tensordot_2/MatMul/pfor/strided_slice:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: k
)loop_body/Tensordot_2/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :?
'loop_body/Tensordot_2/MatMul/pfor/EqualEqual-loop_body/Tensordot_2/MatMul/pfor/Minimum:z:02loop_body/Tensordot_2/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: 
*loop_body/Tensordot_2/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          
*loop_body/Tensordot_2/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          ?
(loop_body/Tensordot_2/MatMul/pfor/SelectSelect+loop_body/Tensordot_2/MatMul/pfor/Equal:z:03loop_body/Tensordot_2/MatMul/pfor/Select/t:output:03loop_body/Tensordot_2/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_2StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_3StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_4StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
'loop_body/Tensordot_2/MatMul/pfor/stackPack:loop_body/Tensordot_2/MatMul/pfor/strided_slice_2:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_3:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
+loop_body/Tensordot_2/MatMul/pfor/transpose	Transpose5loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:01loop_body/Tensordot_2/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'????????????????????????????
)loop_body/Tensordot_2/MatMul/pfor/ReshapeReshape/loop_body/Tensordot_2/MatMul/pfor/transpose:y:00loop_body/Tensordot_2/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:???????????
)loop_body/Tensordot_2/MatMul/pfor/Shape_1Shape2loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:s
1loop_body/Tensordot_2/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
'loop_body/Tensordot_2/MatMul/pfor/splitSplit:loop_body/Tensordot_2/MatMul/pfor/split/split_dim:output:02loop_body/Tensordot_2/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_splitt
1loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB v
3loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
+loop_body/Tensordot_2/MatMul/pfor/Reshape_1Reshape0loop_body/Tensordot_2/MatMul/pfor/split:output:0<loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: t
1loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB v
3loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
+loop_body/Tensordot_2/MatMul/pfor/Reshape_2Reshape0loop_body/Tensordot_2/MatMul/pfor/split:output:1<loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: t
1loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB v
3loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
+loop_body/Tensordot_2/MatMul/pfor/Reshape_3Reshape0loop_body/Tensordot_2/MatMul/pfor/split:output:2<loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ?
%loop_body/Tensordot_2/MatMul/pfor/mulMul4loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ?
1loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shapePack4loop_body/Tensordot_2/MatMul/pfor/Reshape_1:output:0)loop_body/Tensordot_2/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:?
+loop_body/Tensordot_2/MatMul/pfor/Reshape_4Reshape2loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0:loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:???????????????????
(loop_body/Tensordot_2/MatMul/pfor/MatMulMatMul&loop_body/Tensordot_2/Reshape:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:?????????~
3loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
??????????
1loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shapePack<loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:?
+loop_body/Tensordot_2/MatMul/pfor/Reshape_5Reshape2loop_body/Tensordot_2/MatMul/pfor/MatMul:product:0:loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'????????????????????????????
2loop_body/Tensordot_2/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
-loop_body/Tensordot_2/MatMul/pfor/transpose_1	Transpose4loop_body/Tensordot_2/MatMul/pfor/Reshape_5:output:0;loop_body/Tensordot_2/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????h
&loop_body/Tensordot_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
!loop_body/Tensordot_2/pfor/concatConcatV2pfor/Reshape:output:0$loop_body/Tensordot_2/shape:output:0/loop_body/Tensordot_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
"loop_body/Tensordot_2/pfor/ReshapeReshape1loop_body/Tensordot_2/MatMul/pfor/transpose_1:y:0*loop_body/Tensordot_2/pfor/concat:output:0*
T0*(
_output_shapes
:??????????Y
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
value	B :?
loop_body/add/pfor/addAddV2"loop_body/add/pfor/Rank_1:output:0!loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: ?
loop_body/add/pfor/MaximumMaximumloop_body/add/pfor/add:z:0 loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: s
loop_body/add/pfor/ShapeShape+loop_body/Tensordot_2/pfor/Reshape:output:0*
T0*
_output_shapes
:?
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
:*
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
:?
loop_body/add/pfor/Reshape_1Reshape+loop_body/Tensordot_2/pfor/Reshape:output:0"loop_body/add/pfor/concat:output:0*
T0*(
_output_shapes
:???????????
loop_body/add/pfor/AddV2AddV2%loop_body/add/pfor/Reshape_1:output:0$loop_body/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   {
ReshapeReshapeloop_body/add/pfor/AddV2:z:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????W
SoftmaxSoftmaxReshape:output:0*
T0*(
_output_shapes
:??????????a
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^loop_body/ReadVariableOp^loop_body/ReadVariableOp_1^loop_body/ReadVariableOp_2^loop_body/add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@: : : : 24
loop_body/ReadVariableOploop_body/ReadVariableOp28
loop_body/ReadVariableOp_1loop_body/ReadVariableOp_128
loop_body/ReadVariableOp_2loop_body/ReadVariableOp_22<
loop_body/add/ReadVariableOploop_body/add/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
-__inference_sequential_1_layer_call_fn_413861

inputs!
unknown:
	unknown_0:#
	unknown_1:#
	unknown_2:
	unknown_3:#
	unknown_4:#
	unknown_5:#
	unknown_6:#
	unknown_7:
	unknown_8:	?
	unknown_9:%

unknown_10:? 

unknown_11: 

unknown_12:

unknown_13:%

unknown_14:?

unknown_15:

unknown_16:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout	
2*
_collective_manager_ids
 *0
_output_shapes
:??????????: : : : *4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_413197p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????22: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????22
 
_user_specified_nameinputs
?
b
D__inference_dropout1_layer_call_and_return_conditional_losses_412879

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_412324

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
?9
?
E__inference_conv2dmpo_layer_call_and_return_conditional_losses_415796

inputs1
readvariableop_resource:3
readvariableop_1_resource:/
!reshape_2_readvariableop_resource:
identity??ReadVariableOp?ReadVariableOp_1?Reshape_2/ReadVariableOpn
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0r
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*&
_output_shapes
:*
dtype0q
Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Tensordot/transpose	TransposeReadVariableOp_1:value:0!Tensordot/transpose/perm:output:0*
T0*&
_output_shapes
:h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
Tensordot/ReshapeReshapeTensordot/transpose:y:0 Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:s
Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Tensordot/transpose_1	TransposeReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*&
_output_shapes
:j
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:}
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*
_output_shapes

:p
Tensordot/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  ?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*.
_output_shapes
:o
transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   |
	transpose	TransposeTensordot:output:0transpose/perm:output:0*
T0*.
_output_shapes
:q
transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   {
transpose_1	Transposetranspose:y:0transpose_1/perm:output:0*
T0*.
_output_shapes
:f
ShapeConst*
_output_shapes
:*
dtype0*-
value$B""                  ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskO
ConstConst*
_output_shapes
:*
dtype0*
valueB: U
ProdProdstrided_slice:output:0Const:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskT
concat/values_1PackProd:output:0*
N*
T0*
_output_shapes
:V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2strided_slice_1:output:0concat/values_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapetranspose_1:y:0concat:output:0*
T0**
_output_shapes
:m
transpose_2/permConst*
_output_shapes
:*
dtype0*)
value B"                z
transpose_2	TransposeReshape:output:0transpose_2/perm:output:0*
T0**
_output_shapes
:d
Shape_1Const*
_output_shapes
:*
dtype0*)
value B"               _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskQ
Const_1Const*
_output_shapes
:*
dtype0*
valueB: [
Prod_1Prodstrided_slice_2:output:0Const_1:output:0*
T0*
_output_shapes
: _
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskX
concat_1/values_1PackProd_1:output:0*
N*
T0*
_output_shapes
:X
concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concat_1ConcatV2strided_slice_3:output:0concat_1/values_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:i
	Reshape_1Reshapetranspose_2:y:0concat_1:output:0*
T0*&
_output_shapes
:?
Conv2DConv2DinputsReshape_1:output:0*
T0*/
_output_shapes
:?????????//*
paddingSAME*
strides
v
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0`
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      y
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0*
_output_shapes

:k
addAddV2Conv2D:output:0Reshape_2:output:0*
T0*/
_output_shapes
:?????????//O
ReluReluadd:z:0*
T0*/
_output_shapes
:?????????//i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????//?
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^Reshape_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????//: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_124
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:W S
/
_output_shapes
:?????????//
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_412299

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
?^
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_413696
input_12'
conv2d_413615:
conv2d_413617:*
conv2dmpo_413620:*
conv2dmpo_413622:
conv2dmpo_413624:,
conv2dmpo_1_413636:,
conv2dmpo_1_413638:,
conv2dmpo_1_413640:,
conv2dmpo_1_413642:!
conv2dmpo_1_413644:	?&
fruit_tn1_413656:+
fruit_tn1_413658:?&
fruit_tn1_413660:&
fruit_tn1_413662:"
fruit_tn2_413674:+
fruit_tn2_413676:?"
fruit_tn2_413678:
fruit_tn2_413680:	?
identity

identity_1

identity_2

identity_3

identity_4??conv2d/StatefulPartitionedCall?!conv2dmpo/StatefulPartitionedCall?#conv2dmpo_1/StatefulPartitionedCall?!fruit_tn1/StatefulPartitionedCall?!fruit_tn2/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_12conv2d_413615conv2d_413617*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_412370?
!conv2dmpo/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2dmpo_413620conv2dmpo_413622conv2dmpo_413624*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2dmpo_layer_call_and_return_conditional_losses_412438?
-conv2dmpo/ActivityRegularizer/PartitionedCallPartitionedCall*conv2dmpo/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *:
f5R3
1__inference_conv2dmpo_activity_regularizer_412290}
#conv2dmpo/ActivityRegularizer/ShapeShape*conv2dmpo/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1conv2dmpo/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3conv2dmpo/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3conv2dmpo/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+conv2dmpo/ActivityRegularizer/strided_sliceStridedSlice,conv2dmpo/ActivityRegularizer/Shape:output:0:conv2dmpo/ActivityRegularizer/strided_slice/stack:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_1:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"conv2dmpo/ActivityRegularizer/CastCast4conv2dmpo/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%conv2dmpo/ActivityRegularizer/truedivRealDiv6conv2dmpo/ActivityRegularizer/PartitionedCall:output:0&conv2dmpo/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
max_pooling2d/PartitionedCallPartitionedCall*conv2dmpo/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_412299?
#conv2dmpo_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2dmpo_1_413636conv2dmpo_1_413638conv2dmpo_1_413640conv2dmpo_1_413642conv2dmpo_1_413644*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2dmpo_1_layer_call_and_return_conditional_losses_412543?
/conv2dmpo_1/ActivityRegularizer/PartitionedCallPartitionedCall,conv2dmpo_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *<
f7R5
3__inference_conv2dmpo_1_activity_regularizer_412315?
%conv2dmpo_1/ActivityRegularizer/ShapeShape,conv2dmpo_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:}
3conv2dmpo_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5conv2dmpo_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5conv2dmpo_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-conv2dmpo_1/ActivityRegularizer/strided_sliceStridedSlice.conv2dmpo_1/ActivityRegularizer/Shape:output:0<conv2dmpo_1/ActivityRegularizer/strided_slice/stack:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_1:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
$conv2dmpo_1/ActivityRegularizer/CastCast6conv2dmpo_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
'conv2dmpo_1/ActivityRegularizer/truedivRealDiv8conv2dmpo_1/ActivityRegularizer/PartitionedCall:output:0(conv2dmpo_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
max_pooling2d_1/PartitionedCallPartitionedCall,conv2dmpo_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_412324?
!fruit_tn1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0fruit_tn1_413656fruit_tn1_413658fruit_tn1_413660fruit_tn1_413662*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_fruit_tn1_layer_call_and_return_conditional_losses_412856?
-fruit_tn1/ActivityRegularizer/PartitionedCallPartitionedCall*fruit_tn1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *:
f5R3
1__inference_fruit_tn1_activity_regularizer_412340}
#fruit_tn1/ActivityRegularizer/ShapeShape*fruit_tn1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1fruit_tn1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3fruit_tn1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3fruit_tn1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+fruit_tn1/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn1/ActivityRegularizer/Shape:output:0:fruit_tn1/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"fruit_tn1/ActivityRegularizer/CastCast4fruit_tn1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%fruit_tn1/ActivityRegularizer/truedivRealDiv6fruit_tn1/ActivityRegularizer/PartitionedCall:output:0&fruit_tn1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
dropout1/PartitionedCallPartitionedCall*fruit_tn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout1_layer_call_and_return_conditional_losses_412879?
!fruit_tn2/StatefulPartitionedCallStatefulPartitionedCall!dropout1/PartitionedCall:output:0fruit_tn2_413674fruit_tn2_413676fruit_tn2_413678fruit_tn2_413680*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_fruit_tn2_layer_call_and_return_conditional_losses_413174?
-fruit_tn2/ActivityRegularizer/PartitionedCallPartitionedCall*fruit_tn2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *:
f5R3
1__inference_fruit_tn2_activity_regularizer_412353}
#fruit_tn2/ActivityRegularizer/ShapeShape*fruit_tn2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1fruit_tn2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3fruit_tn2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3fruit_tn2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+fruit_tn2/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn2/ActivityRegularizer/Shape:output:0:fruit_tn2/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"fruit_tn2/ActivityRegularizer/CastCast4fruit_tn2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%fruit_tn2/ActivityRegularizer/truedivRealDiv6fruit_tn2/ActivityRegularizer/PartitionedCall:output:0&fruit_tn2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: z
IdentityIdentity*fruit_tn2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????i

Identity_1Identity)conv2dmpo/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: k

Identity_2Identity+conv2dmpo_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_3Identity)fruit_tn1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_4Identity)fruit_tn2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^conv2d/StatefulPartitionedCall"^conv2dmpo/StatefulPartitionedCall$^conv2dmpo_1/StatefulPartitionedCall"^fruit_tn1/StatefulPartitionedCall"^fruit_tn2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????22: : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2F
!conv2dmpo/StatefulPartitionedCall!conv2dmpo/StatefulPartitionedCall2J
#conv2dmpo_1/StatefulPartitionedCall#conv2dmpo_1/StatefulPartitionedCall2F
!fruit_tn1/StatefulPartitionedCall!fruit_tn1/StatefulPartitionedCall2F
!fruit_tn2/StatefulPartitionedCall!fruit_tn2/StatefulPartitionedCall:Y U
/
_output_shapes
:?????????22
"
_user_specified_name
input_12
?
?
K__inference_conv2dmpo_1_layer_call_and_return_all_conditional_losses_415625

inputs!
unknown:#
	unknown_0:#
	unknown_1:#
	unknown_2:
	unknown_3:	?
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2dmpo_1_layer_call_and_return_conditional_losses_412543?
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
GPU2*0J 8? *<
f7R5
3__inference_conv2dmpo_1_activity_regularizer_412315x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????X

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
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_fruit_tn1_layer_call_and_return_all_conditional_losses_415671

inputs
unknown:$
	unknown_0:?
	unknown_1:
	unknown_2:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_fruit_tn1_layer_call_and_return_conditional_losses_412856?
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
GPU2*0J 8? *:
f5R3
1__inference_fruit_tn1_activity_regularizer_412340o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@X

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
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?3
"__inference__traced_restore_416982
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias:E
+assignvariableop_2_conv2dmpo_end_node_first:D
*assignvariableop_3_conv2dmpo_end_node_last:/
!assignvariableop_4_conv2dmpo_bias:G
-assignvariableop_5_conv2dmpo_1_end_node_first:F
,assignvariableop_6_conv2dmpo_1_middle_node_0:F
,assignvariableop_7_conv2dmpo_1_middle_node_1:F
,assignvariableop_8_conv2dmpo_1_end_node_last:2
#assignvariableop_9_conv2dmpo_1_bias:	?+
assignvariableop_10_a:0
assignvariableop_11_b:?+
assignvariableop_12_c:.
assignvariableop_13_bias:)
assignvariableop_14_a_1:2
assignvariableop_15_b_1:?)
assignvariableop_16_c_1:)
assignvariableop_17_bias_1:	?'
assignvariableop_18_adam_iter:	 )
assignvariableop_19_adam_beta_1: )
assignvariableop_20_adam_beta_2: (
assignvariableop_21_adam_decay: #
assignvariableop_22_total: #
assignvariableop_23_count: 0
"assignvariableop_24_true_positives:1
#assignvariableop_25_false_positives:B
(assignvariableop_26_adam_conv2d_kernel_m:4
&assignvariableop_27_adam_conv2d_bias_m:M
3assignvariableop_28_adam_conv2dmpo_end_node_first_m:L
2assignvariableop_29_adam_conv2dmpo_end_node_last_m:7
)assignvariableop_30_adam_conv2dmpo_bias_m:O
5assignvariableop_31_adam_conv2dmpo_1_end_node_first_m:N
4assignvariableop_32_adam_conv2dmpo_1_middle_node_0_m:N
4assignvariableop_33_adam_conv2dmpo_1_middle_node_1_m:N
4assignvariableop_34_adam_conv2dmpo_1_end_node_last_m::
+assignvariableop_35_adam_conv2dmpo_1_bias_m:	?2
assignvariableop_36_adam_a_m:7
assignvariableop_37_adam_b_m:?2
assignvariableop_38_adam_c_m:5
assignvariableop_39_adam_bias_m:0
assignvariableop_40_adam_a_m_1:9
assignvariableop_41_adam_b_m_1:?0
assignvariableop_42_adam_c_m_1:0
!assignvariableop_43_adam_bias_m_1:	?B
(assignvariableop_44_adam_conv2d_kernel_v:4
&assignvariableop_45_adam_conv2d_bias_v:M
3assignvariableop_46_adam_conv2dmpo_end_node_first_v:L
2assignvariableop_47_adam_conv2dmpo_end_node_last_v:7
)assignvariableop_48_adam_conv2dmpo_bias_v:O
5assignvariableop_49_adam_conv2dmpo_1_end_node_first_v:N
4assignvariableop_50_adam_conv2dmpo_1_middle_node_0_v:N
4assignvariableop_51_adam_conv2dmpo_1_middle_node_1_v:N
4assignvariableop_52_adam_conv2dmpo_1_end_node_last_v::
+assignvariableop_53_adam_conv2dmpo_1_bias_v:	?2
assignvariableop_54_adam_a_v:7
assignvariableop_55_adam_b_v:?2
assignvariableop_56_adam_c_v:5
assignvariableop_57_adam_bias_v:0
assignvariableop_58_adam_a_v_1:9
assignvariableop_59_adam_b_v_1:?0
assignvariableop_60_adam_c_v_1:0
!assignvariableop_61_adam_bias_v_1:	?E
+assignvariableop_62_adam_conv2d_kernel_vhat:7
)assignvariableop_63_adam_conv2d_bias_vhat:P
6assignvariableop_64_adam_conv2dmpo_end_node_first_vhat:O
5assignvariableop_65_adam_conv2dmpo_end_node_last_vhat::
,assignvariableop_66_adam_conv2dmpo_bias_vhat:R
8assignvariableop_67_adam_conv2dmpo_1_end_node_first_vhat:Q
7assignvariableop_68_adam_conv2dmpo_1_middle_node_0_vhat:Q
7assignvariableop_69_adam_conv2dmpo_1_middle_node_1_vhat:Q
7assignvariableop_70_adam_conv2dmpo_1_end_node_last_vhat:=
.assignvariableop_71_adam_conv2dmpo_1_bias_vhat:	?5
assignvariableop_72_adam_a_vhat::
assignvariableop_73_adam_b_vhat:?5
assignvariableop_74_adam_c_vhat:8
"assignvariableop_75_adam_bias_vhat:3
!assignvariableop_76_adam_a_vhat_1:<
!assignvariableop_77_adam_b_vhat_1:?3
!assignvariableop_78_adam_c_vhat_1:3
$assignvariableop_79_adam_bias_vhat_1:	?
identity_81??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_9?0
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*?0
value?0B?0QB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/end_node_first/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-1/end_node_last/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/end_node_first/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-2/middle_node_0/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-2/middle_node_1/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-2/end_node_last/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/a_var/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/b_var/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/c_var/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/a_var/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/b_var/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/c_var/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/end_node_first/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/end_node_last/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/end_node_first/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/middle_node_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/middle_node_1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/end_node_last/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/a_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/b_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/c_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/a_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/b_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/c_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/end_node_first/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/end_node_last/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/end_node_first/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/middle_node_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/middle_node_1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/end_node_last/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/a_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/b_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/c_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/a_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/b_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/c_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-1/end_node_first/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/end_node_last/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-2/end_node_first/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/middle_node_0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/middle_node_1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/end_node_last/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/a_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/b_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/c_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/a_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/b_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/c_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*?
value?B?QB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*_
dtypesU
S2Q	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp+assignvariableop_2_conv2dmpo_end_node_firstIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp*assignvariableop_3_conv2dmpo_end_node_lastIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_conv2dmpo_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp-assignvariableop_5_conv2dmpo_1_end_node_firstIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp,assignvariableop_6_conv2dmpo_1_middle_node_0Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp,assignvariableop_7_conv2dmpo_1_middle_node_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp,assignvariableop_8_conv2dmpo_1_end_node_lastIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_conv2dmpo_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_aIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_bIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_cIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_a_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_b_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_c_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_bias_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_beta_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_decayIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp"assignvariableop_24_true_positivesIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp#assignvariableop_25_false_positivesIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp&assignvariableop_27_adam_conv2d_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp3assignvariableop_28_adam_conv2dmpo_end_node_first_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp2assignvariableop_29_adam_conv2dmpo_end_node_last_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2dmpo_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp5assignvariableop_31_adam_conv2dmpo_1_end_node_first_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp4assignvariableop_32_adam_conv2dmpo_1_middle_node_0_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp4assignvariableop_33_adam_conv2dmpo_1_middle_node_1_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adam_conv2dmpo_1_end_node_last_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2dmpo_1_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOpassignvariableop_36_adam_a_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOpassignvariableop_37_adam_b_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOpassignvariableop_38_adam_c_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOpassignvariableop_39_adam_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOpassignvariableop_40_adam_a_m_1Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOpassignvariableop_41_adam_b_m_1Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOpassignvariableop_42_adam_c_m_1Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp!assignvariableop_43_adam_bias_m_1Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv2d_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp&assignvariableop_45_adam_conv2d_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp3assignvariableop_46_adam_conv2dmpo_end_node_first_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp2assignvariableop_47_adam_conv2dmpo_end_node_last_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2dmpo_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp5assignvariableop_49_adam_conv2dmpo_1_end_node_first_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp4assignvariableop_50_adam_conv2dmpo_1_middle_node_0_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp4assignvariableop_51_adam_conv2dmpo_1_middle_node_1_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp4assignvariableop_52_adam_conv2dmpo_1_end_node_last_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_conv2dmpo_1_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOpassignvariableop_54_adam_a_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOpassignvariableop_55_adam_b_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOpassignvariableop_56_adam_c_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOpassignvariableop_57_adam_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOpassignvariableop_58_adam_a_v_1Identity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOpassignvariableop_59_adam_b_v_1Identity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOpassignvariableop_60_adam_c_v_1Identity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp!assignvariableop_61_adam_bias_v_1Identity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp+assignvariableop_62_adam_conv2d_kernel_vhatIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp)assignvariableop_63_adam_conv2d_bias_vhatIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_conv2dmpo_end_node_first_vhatIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp5assignvariableop_65_adam_conv2dmpo_end_node_last_vhatIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp,assignvariableop_66_adam_conv2dmpo_bias_vhatIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp8assignvariableop_67_adam_conv2dmpo_1_end_node_first_vhatIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp7assignvariableop_68_adam_conv2dmpo_1_middle_node_0_vhatIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp7assignvariableop_69_adam_conv2dmpo_1_middle_node_1_vhatIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp7assignvariableop_70_adam_conv2dmpo_1_end_node_last_vhatIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp.assignvariableop_71_adam_conv2dmpo_1_bias_vhatIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOpassignvariableop_72_adam_a_vhatIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOpassignvariableop_73_adam_b_vhatIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOpassignvariableop_74_adam_c_vhatIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOp"assignvariableop_75_adam_bias_vhatIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOp!assignvariableop_76_adam_a_vhat_1Identity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOp!assignvariableop_77_adam_b_vhat_1Identity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOp!assignvariableop_78_adam_c_vhat_1Identity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOp$assignvariableop_79_adam_bias_vhat_1Identity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_80Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_81IdentityIdentity_80:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_81Identity_81:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

?
B__inference_conv2d_layer_call_and_return_conditional_losses_415559

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????//*
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
:?????????//g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????//w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????22
 
_user_specified_nameinputs
?
?
I__inference_conv2dmpo_layer_call_and_return_all_conditional_losses_415583

inputs!
unknown:#
	unknown_0:
	unknown_1:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2dmpo_layer_call_and_return_conditional_losses_412438?
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
GPU2*0J 8? *:
f5R3
1__inference_conv2dmpo_activity_regularizer_412290w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????//X

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
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????//: : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????//
 
_user_specified_nameinputs
?
b
)__inference_dropout1_layer_call_fn_415681

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout1_layer_call_and_return_conditional_losses_413290o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
E
)__inference_dropout1_layer_call_fn_415676

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout1_layer_call_and_return_conditional_losses_412879`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_415497

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:;
!conv2dmpo_readvariableop_resource:=
#conv2dmpo_readvariableop_1_resource:9
+conv2dmpo_reshape_2_readvariableop_resource:=
#conv2dmpo_1_readvariableop_resource:?
%conv2dmpo_1_readvariableop_1_resource:?
%conv2dmpo_1_readvariableop_2_resource:?
%conv2dmpo_1_readvariableop_3_resource:<
-conv2dmpo_1_reshape_2_readvariableop_resource:	?A
+fruit_tn1_loop_body_readvariableop_resource:H
-fruit_tn1_loop_body_readvariableop_1_resource:?C
-fruit_tn1_loop_body_readvariableop_2_resource:E
/fruit_tn1_loop_body_add_readvariableop_resource:=
+fruit_tn2_loop_body_readvariableop_resource:H
-fruit_tn2_loop_body_readvariableop_1_resource:??
-fruit_tn2_loop_body_readvariableop_2_resource:>
/fruit_tn2_loop_body_add_readvariableop_resource:	?
identity

identity_1

identity_2

identity_3

identity_4??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2dmpo/ReadVariableOp?conv2dmpo/ReadVariableOp_1?"conv2dmpo/Reshape_2/ReadVariableOp?conv2dmpo_1/ReadVariableOp?conv2dmpo_1/ReadVariableOp_1?conv2dmpo_1/ReadVariableOp_2?conv2dmpo_1/ReadVariableOp_3?$conv2dmpo_1/Reshape_2/ReadVariableOp?"fruit_tn1/loop_body/ReadVariableOp?$fruit_tn1/loop_body/ReadVariableOp_1?$fruit_tn1/loop_body/ReadVariableOp_2?&fruit_tn1/loop_body/add/ReadVariableOp?"fruit_tn2/loop_body/ReadVariableOp?$fruit_tn2/loop_body/ReadVariableOp_1?$fruit_tn2/loop_body/ReadVariableOp_2?&fruit_tn2/loop_body/add/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????//*
paddingVALID*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????//?
conv2dmpo/ReadVariableOpReadVariableOp!conv2dmpo_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2dmpo/ReadVariableOp_1ReadVariableOp#conv2dmpo_readvariableop_1_resource*&
_output_shapes
:*
dtype0{
"conv2dmpo/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2dmpo/Tensordot/transpose	Transpose conv2dmpo/ReadVariableOp:value:0+conv2dmpo/Tensordot/transpose/perm:output:0*
T0*&
_output_shapes
:r
!conv2dmpo/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
conv2dmpo/Tensordot/ReshapeReshape!conv2dmpo/Tensordot/transpose:y:0*conv2dmpo/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:}
$conv2dmpo/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2dmpo/Tensordot/transpose_1	Transpose"conv2dmpo/ReadVariableOp_1:value:0-conv2dmpo/Tensordot/transpose_1/perm:output:0*
T0*&
_output_shapes
:t
#conv2dmpo/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
conv2dmpo/Tensordot/Reshape_1Reshape#conv2dmpo/Tensordot/transpose_1:y:0,conv2dmpo/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:?
conv2dmpo/Tensordot/MatMulMatMul$conv2dmpo/Tensordot/Reshape:output:0&conv2dmpo/Tensordot/Reshape_1:output:0*
T0*
_output_shapes

:z
conv2dmpo/Tensordot/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  ?
conv2dmpo/TensordotReshape$conv2dmpo/Tensordot/MatMul:product:0"conv2dmpo/Tensordot/shape:output:0*
T0*.
_output_shapes
:y
conv2dmpo/transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
conv2dmpo/transpose	Transposeconv2dmpo/Tensordot:output:0!conv2dmpo/transpose/perm:output:0*
T0*.
_output_shapes
:{
conv2dmpo/transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
conv2dmpo/transpose_1	Transposeconv2dmpo/transpose:y:0#conv2dmpo/transpose_1/perm:output:0*
T0*.
_output_shapes
:p
conv2dmpo/ShapeConst*
_output_shapes
:*
dtype0*-
value$B""                  g
conv2dmpo/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
conv2dmpo/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
conv2dmpo/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2dmpo/strided_sliceStridedSliceconv2dmpo/Shape:output:0&conv2dmpo/strided_slice/stack:output:0(conv2dmpo/strided_slice/stack_1:output:0(conv2dmpo/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskY
conv2dmpo/ConstConst*
_output_shapes
:*
dtype0*
valueB: s
conv2dmpo/ProdProd conv2dmpo/strided_slice:output:0conv2dmpo/Const:output:0*
T0*
_output_shapes
: i
conv2dmpo/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!conv2dmpo/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!conv2dmpo/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2dmpo/strided_slice_1StridedSliceconv2dmpo/Shape:output:0(conv2dmpo/strided_slice_1/stack:output:0*conv2dmpo/strided_slice_1/stack_1:output:0*conv2dmpo/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
conv2dmpo/concat/values_1Packconv2dmpo/Prod:output:0*
N*
T0*
_output_shapes
:`
conv2dmpo/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv2dmpo/concatConcatV2"conv2dmpo/strided_slice_1:output:0"conv2dmpo/concat/values_1:output:0conv2dmpo/concat/axis:output:0*
N*
T0*
_output_shapes
:?
conv2dmpo/ReshapeReshapeconv2dmpo/transpose_1:y:0conv2dmpo/concat:output:0*
T0**
_output_shapes
:w
conv2dmpo/transpose_2/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
conv2dmpo/transpose_2	Transposeconv2dmpo/Reshape:output:0#conv2dmpo/transpose_2/perm:output:0*
T0**
_output_shapes
:n
conv2dmpo/Shape_1Const*
_output_shapes
:*
dtype0*)
value B"               i
conv2dmpo/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:k
!conv2dmpo/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: k
!conv2dmpo/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2dmpo/strided_slice_2StridedSliceconv2dmpo/Shape_1:output:0(conv2dmpo/strided_slice_2/stack:output:0*conv2dmpo/strided_slice_2/stack_1:output:0*conv2dmpo/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask[
conv2dmpo/Const_1Const*
_output_shapes
:*
dtype0*
valueB: y
conv2dmpo/Prod_1Prod"conv2dmpo/strided_slice_2:output:0conv2dmpo/Const_1:output:0*
T0*
_output_shapes
: i
conv2dmpo/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!conv2dmpo/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!conv2dmpo/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2dmpo/strided_slice_3StridedSliceconv2dmpo/Shape_1:output:0(conv2dmpo/strided_slice_3/stack:output:0*conv2dmpo/strided_slice_3/stack_1:output:0*conv2dmpo/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskl
conv2dmpo/concat_1/values_1Packconv2dmpo/Prod_1:output:0*
N*
T0*
_output_shapes
:b
conv2dmpo/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv2dmpo/concat_1ConcatV2"conv2dmpo/strided_slice_3:output:0$conv2dmpo/concat_1/values_1:output:0 conv2dmpo/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
conv2dmpo/Reshape_1Reshapeconv2dmpo/transpose_2:y:0conv2dmpo/concat_1:output:0*
T0*&
_output_shapes
:?
conv2dmpo/Conv2DConv2Dconv2d/BiasAdd:output:0conv2dmpo/Reshape_1:output:0*
T0*/
_output_shapes
:?????????//*
paddingSAME*
strides
?
"conv2dmpo/Reshape_2/ReadVariableOpReadVariableOp+conv2dmpo_reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0j
conv2dmpo/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
conv2dmpo/Reshape_2Reshape*conv2dmpo/Reshape_2/ReadVariableOp:value:0"conv2dmpo/Reshape_2/shape:output:0*
T0*
_output_shapes

:?
conv2dmpo/addAddV2conv2dmpo/Conv2D:output:0conv2dmpo/Reshape_2:output:0*
T0*/
_output_shapes
:?????????//c
conv2dmpo/ReluReluconv2dmpo/add:z:0*
T0*/
_output_shapes
:?????????//?
$conv2dmpo/ActivityRegularizer/SquareSquareconv2dmpo/Relu:activations:0*
T0*/
_output_shapes
:?????????//|
#conv2dmpo/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2dmpo/ActivityRegularizer/SumSum(conv2dmpo/ActivityRegularizer/Square:y:0,conv2dmpo/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: h
#conv2dmpo/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
!conv2dmpo/ActivityRegularizer/mulMul,conv2dmpo/ActivityRegularizer/mul/x:output:0*conv2dmpo/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: o
#conv2dmpo/ActivityRegularizer/ShapeShapeconv2dmpo/Relu:activations:0*
T0*
_output_shapes
:{
1conv2dmpo/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3conv2dmpo/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3conv2dmpo/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+conv2dmpo/ActivityRegularizer/strided_sliceStridedSlice,conv2dmpo/ActivityRegularizer/Shape:output:0:conv2dmpo/ActivityRegularizer/strided_slice/stack:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_1:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"conv2dmpo/ActivityRegularizer/CastCast4conv2dmpo/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%conv2dmpo/ActivityRegularizer/truedivRealDiv%conv2dmpo/ActivityRegularizer/mul:z:0&conv2dmpo/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
max_pooling2d/MaxPoolMaxPoolconv2dmpo/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
?
conv2dmpo_1/ReadVariableOpReadVariableOp#conv2dmpo_1_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2dmpo_1/ReadVariableOp_1ReadVariableOp%conv2dmpo_1_readvariableop_1_resource*&
_output_shapes
:*
dtype0?
conv2dmpo_1/ReadVariableOp_2ReadVariableOp%conv2dmpo_1_readvariableop_2_resource*&
_output_shapes
:*
dtype0?
conv2dmpo_1/ReadVariableOp_3ReadVariableOp%conv2dmpo_1_readvariableop_3_resource*&
_output_shapes
:*
dtype0}
$conv2dmpo_1/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2dmpo_1/Tensordot/transpose	Transpose$conv2dmpo_1/ReadVariableOp_1:value:0-conv2dmpo_1/Tensordot/transpose/perm:output:0*
T0*&
_output_shapes
:t
#conv2dmpo_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@      ?
conv2dmpo_1/Tensordot/ReshapeReshape#conv2dmpo_1/Tensordot/transpose:y:0,conv2dmpo_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@
&conv2dmpo_1/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2dmpo_1/Tensordot/transpose_1	Transpose"conv2dmpo_1/ReadVariableOp:value:0/conv2dmpo_1/Tensordot/transpose_1/perm:output:0*
T0*&
_output_shapes
:v
%conv2dmpo_1/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
conv2dmpo_1/Tensordot/Reshape_1Reshape%conv2dmpo_1/Tensordot/transpose_1:y:0.conv2dmpo_1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:?
conv2dmpo_1/Tensordot/MatMulMatMul&conv2dmpo_1/Tensordot/Reshape:output:0(conv2dmpo_1/Tensordot/Reshape_1:output:0*
T0*
_output_shapes

:@|
conv2dmpo_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  ?
conv2dmpo_1/TensordotReshape&conv2dmpo_1/Tensordot/MatMul:product:0$conv2dmpo_1/Tensordot/shape:output:0*
T0*.
_output_shapes
:
&conv2dmpo_1/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2dmpo_1/Tensordot_1/transpose	Transpose$conv2dmpo_1/ReadVariableOp_3:value:0/conv2dmpo_1/Tensordot_1/transpose/perm:output:0*
T0*&
_output_shapes
:v
%conv2dmpo_1/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
conv2dmpo_1/Tensordot_1/ReshapeReshape%conv2dmpo_1/Tensordot_1/transpose:y:0.conv2dmpo_1/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:?
(conv2dmpo_1/Tensordot_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
#conv2dmpo_1/Tensordot_1/transpose_1	Transpose$conv2dmpo_1/ReadVariableOp_2:value:01conv2dmpo_1/Tensordot_1/transpose_1/perm:output:0*
T0*&
_output_shapes
:x
'conv2dmpo_1/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   ?
!conv2dmpo_1/Tensordot_1/Reshape_1Reshape'conv2dmpo_1/Tensordot_1/transpose_1:y:00conv2dmpo_1/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:@?
conv2dmpo_1/Tensordot_1/MatMulMatMul(conv2dmpo_1/Tensordot_1/Reshape:output:0*conv2dmpo_1/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes

:@~
conv2dmpo_1/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  ?
conv2dmpo_1/Tensordot_1Reshape(conv2dmpo_1/Tensordot_1/MatMul:product:0&conv2dmpo_1/Tensordot_1/shape:output:0*
T0*.
_output_shapes
:?
&conv2dmpo_1/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
!conv2dmpo_1/Tensordot_2/transpose	Transposeconv2dmpo_1/Tensordot:output:0/conv2dmpo_1/Tensordot_2/transpose/perm:output:0*
T0*.
_output_shapes
:v
%conv2dmpo_1/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?      ?
conv2dmpo_1/Tensordot_2/ReshapeReshape%conv2dmpo_1/Tensordot_2/transpose:y:0.conv2dmpo_1/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes
:	??
(conv2dmpo_1/Tensordot_2/transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
#conv2dmpo_1/Tensordot_2/transpose_1	Transpose conv2dmpo_1/Tensordot_1:output:01conv2dmpo_1/Tensordot_2/transpose_1/perm:output:0*
T0*.
_output_shapes
:x
'conv2dmpo_1/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   ?
!conv2dmpo_1/Tensordot_2/Reshape_1Reshape'conv2dmpo_1/Tensordot_2/transpose_1:y:00conv2dmpo_1/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	??
conv2dmpo_1/Tensordot_2/MatMulMatMul(conv2dmpo_1/Tensordot_2/Reshape:output:0*conv2dmpo_1/Tensordot_2/Reshape_1:output:0*
T0* 
_output_shapes
:
???
conv2dmpo_1/Tensordot_2/shapeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                              ?
conv2dmpo_1/Tensordot_2Reshape(conv2dmpo_1/Tensordot_2/MatMul:product:0&conv2dmpo_1/Tensordot_2/shape:output:0*
T0*>
_output_shapes,
*:(?
conv2dmpo_1/transpose/permConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                         	      ?
conv2dmpo_1/transpose	Transpose conv2dmpo_1/Tensordot_2:output:0#conv2dmpo_1/transpose/perm:output:0*
T0*>
_output_shapes,
*:(?
conv2dmpo_1/transpose_1/permConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                	               ?
conv2dmpo_1/transpose_1	Transposeconv2dmpo_1/transpose:y:0%conv2dmpo_1/transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(?
conv2dmpo_1/ShapeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                              i
conv2dmpo_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:k
!conv2dmpo_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: k
!conv2dmpo_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2dmpo_1/strided_sliceStridedSliceconv2dmpo_1/Shape:output:0(conv2dmpo_1/strided_slice/stack:output:0*conv2dmpo_1/strided_slice/stack_1:output:0*conv2dmpo_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask[
conv2dmpo_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: y
conv2dmpo_1/ProdProd"conv2dmpo_1/strided_slice:output:0conv2dmpo_1/Const:output:0*
T0*
_output_shapes
: k
!conv2dmpo_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#conv2dmpo_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#conv2dmpo_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2dmpo_1/strided_slice_1StridedSliceconv2dmpo_1/Shape:output:0*conv2dmpo_1/strided_slice_1/stack:output:0,conv2dmpo_1/strided_slice_1/stack_1:output:0,conv2dmpo_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskl
conv2dmpo_1/concat/values_1Packconv2dmpo_1/Prod:output:0*
N*
T0*
_output_shapes
:b
conv2dmpo_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv2dmpo_1/concatConcatV2$conv2dmpo_1/strided_slice_1:output:0$conv2dmpo_1/concat/values_1:output:0 conv2dmpo_1/concat/axis:output:0*
N*
T0*
_output_shapes
:?
conv2dmpo_1/ReshapeReshapeconv2dmpo_1/transpose_1:y:0conv2dmpo_1/concat:output:0*
T0*2
_output_shapes 
:?
conv2dmpo_1/transpose_2/permConst*
_output_shapes
:*
dtype0*1
value(B&"                      ?
conv2dmpo_1/transpose_2	Transposeconv2dmpo_1/Reshape:output:0%conv2dmpo_1/transpose_2/perm:output:0*
T0*2
_output_shapes 
:x
conv2dmpo_1/Shape_1Const*
_output_shapes
:*
dtype0*1
value(B&"                     k
!conv2dmpo_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:m
#conv2dmpo_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: m
#conv2dmpo_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2dmpo_1/strided_slice_2StridedSliceconv2dmpo_1/Shape_1:output:0*conv2dmpo_1/strided_slice_2/stack:output:0,conv2dmpo_1/strided_slice_2/stack_1:output:0,conv2dmpo_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask]
conv2dmpo_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
conv2dmpo_1/Prod_1Prod$conv2dmpo_1/strided_slice_2:output:0conv2dmpo_1/Const_1:output:0*
T0*
_output_shapes
: k
!conv2dmpo_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#conv2dmpo_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#conv2dmpo_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2dmpo_1/strided_slice_3StridedSliceconv2dmpo_1/Shape_1:output:0*conv2dmpo_1/strided_slice_3/stack:output:0,conv2dmpo_1/strided_slice_3/stack_1:output:0,conv2dmpo_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
conv2dmpo_1/concat_1/values_1Packconv2dmpo_1/Prod_1:output:0*
N*
T0*
_output_shapes
:d
conv2dmpo_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv2dmpo_1/concat_1ConcatV2$conv2dmpo_1/strided_slice_3:output:0&conv2dmpo_1/concat_1/values_1:output:0"conv2dmpo_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
conv2dmpo_1/Reshape_1Reshapeconv2dmpo_1/transpose_2:y:0conv2dmpo_1/concat_1:output:0*
T0*'
_output_shapes
:??
conv2dmpo_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0conv2dmpo_1/Reshape_1:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
$conv2dmpo_1/Reshape_2/ReadVariableOpReadVariableOp-conv2dmpo_1_reshape_2_readvariableop_resource*
_output_shapes	
:?*
dtype0l
conv2dmpo_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
conv2dmpo_1/Reshape_2Reshape,conv2dmpo_1/Reshape_2/ReadVariableOp:value:0$conv2dmpo_1/Reshape_2/shape:output:0*
T0*
_output_shapes
:	??
conv2dmpo_1/addAddV2conv2dmpo_1/Conv2D:output:0conv2dmpo_1/Reshape_2:output:0*
T0*0
_output_shapes
:??????????h
conv2dmpo_1/ReluReluconv2dmpo_1/add:z:0*
T0*0
_output_shapes
:???????????
&conv2dmpo_1/ActivityRegularizer/SquareSquareconv2dmpo_1/Relu:activations:0*
T0*0
_output_shapes
:??????????~
%conv2dmpo_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
#conv2dmpo_1/ActivityRegularizer/SumSum*conv2dmpo_1/ActivityRegularizer/Square:y:0.conv2dmpo_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: j
%conv2dmpo_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#conv2dmpo_1/ActivityRegularizer/mulMul.conv2dmpo_1/ActivityRegularizer/mul/x:output:0,conv2dmpo_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: s
%conv2dmpo_1/ActivityRegularizer/ShapeShapeconv2dmpo_1/Relu:activations:0*
T0*
_output_shapes
:}
3conv2dmpo_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5conv2dmpo_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5conv2dmpo_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-conv2dmpo_1/ActivityRegularizer/strided_sliceStridedSlice.conv2dmpo_1/ActivityRegularizer/Shape:output:0<conv2dmpo_1/ActivityRegularizer/strided_slice/stack:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_1:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
$conv2dmpo_1/ActivityRegularizer/CastCast6conv2dmpo_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
'conv2dmpo_1/ActivityRegularizer/truedivRealDiv'conv2dmpo_1/ActivityRegularizer/mul:z:0(conv2dmpo_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
max_pooling2d_1/MaxPoolMaxPoolconv2dmpo_1/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
_
fruit_tn1/ShapeShape max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:g
fruit_tn1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
fruit_tn1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
fruit_tn1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
fruit_tn1/strided_sliceStridedSlicefruit_tn1/Shape:output:0&fruit_tn1/strided_slice/stack:output:0(fruit_tn1/strided_slice/stack_1:output:0(fruit_tn1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
fruit_tn1/Rank/packedPack fruit_tn1/strided_slice:output:0*
N*
T0*
_output_shapes
:P
fruit_tn1/RankConst*
_output_shapes
: *
dtype0*
value	B :W
fruit_tn1/range/startConst*
_output_shapes
: *
dtype0*
value	B : W
fruit_tn1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
fruit_tn1/rangeRangefruit_tn1/range/start:output:0fruit_tn1/Rank:output:0fruit_tn1/range/delta:output:0*
_output_shapes
:k
fruit_tn1/Max/inputPack fruit_tn1/strided_slice:output:0*
N*
T0*
_output_shapes
:m
fruit_tn1/MaxMaxfruit_tn1/Max/input:output:0fruit_tn1/range:output:0*
T0*
_output_shapes
: r
0fruit_tn1/loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : ?
*fruit_tn1/loop_body/PlaceholderWithDefaultPlaceholderWithDefault9fruit_tn1/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: i
fruit_tn1/loop_body/ShapeShape max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:q
'fruit_tn1/loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)fruit_tn1/loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)fruit_tn1/loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!fruit_tn1/loop_body/strided_sliceStridedSlice"fruit_tn1/loop_body/Shape:output:00fruit_tn1/loop_body/strided_slice/stack:output:02fruit_tn1/loop_body/strided_slice/stack_1:output:02fruit_tn1/loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
fruit_tn1/loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :?
fruit_tn1/loop_body/GreaterGreater*fruit_tn1/loop_body/strided_slice:output:0&fruit_tn1/loop_body/Greater/y:output:0*
T0*
_output_shapes
: `
fruit_tn1/loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : ?
fruit_tn1/loop_body/SelectV2SelectV2fruit_tn1/loop_body/Greater:z:03fruit_tn1/loop_body/PlaceholderWithDefault:output:0'fruit_tn1/loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: c
!fruit_tn1/loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
fruit_tn1/loop_body/GatherV2GatherV2 max_pooling2d_1/MaxPool:output:0%fruit_tn1/loop_body/SelectV2:output:0*fruit_tn1/loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:??
"fruit_tn1/loop_body/ReadVariableOpReadVariableOp+fruit_tn1_loop_body_readvariableop_resource*"
_output_shapes
:*
dtype0?
$fruit_tn1/loop_body/ReadVariableOp_1ReadVariableOp-fruit_tn1_loop_body_readvariableop_1_resource*'
_output_shapes
:?*
dtype0?
$fruit_tn1/loop_body/ReadVariableOp_2ReadVariableOp-fruit_tn1_loop_body_readvariableop_2_resource*"
_output_shapes
:*
dtype0?
,fruit_tn1/loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
'fruit_tn1/loop_body/Tensordot/transpose	Transpose%fruit_tn1/loop_body/GatherV2:output:05fruit_tn1/loop_body/Tensordot/transpose/perm:output:0*
T0*#
_output_shapes
:?|
+fruit_tn1/loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
%fruit_tn1/loop_body/Tensordot/ReshapeReshape+fruit_tn1/loop_body/Tensordot/transpose:y:04fruit_tn1/loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	??
.fruit_tn1/loop_body/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
)fruit_tn1/loop_body/Tensordot/transpose_1	Transpose*fruit_tn1/loop_body/ReadVariableOp:value:07fruit_tn1/loop_body/Tensordot/transpose_1/perm:output:0*
T0*"
_output_shapes
:~
-fruit_tn1/loop_body/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
'fruit_tn1/loop_body/Tensordot/Reshape_1Reshape-fruit_tn1/loop_body/Tensordot/transpose_1:y:06fruit_tn1/loop_body/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:?
$fruit_tn1/loop_body/Tensordot/MatMulMatMul.fruit_tn1/loop_body/Tensordot/Reshape:output:00fruit_tn1/loop_body/Tensordot/Reshape_1:output:0*
T0*
_output_shapes
:	?|
#fruit_tn1/loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            ?
fruit_tn1/loop_body/TensordotReshape.fruit_tn1/loop_body/Tensordot/MatMul:product:0,fruit_tn1/loop_body/Tensordot/shape:output:0*
T0*'
_output_shapes
:??
.fruit_tn1/loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
)fruit_tn1/loop_body/Tensordot_1/transpose	Transpose,fruit_tn1/loop_body/ReadVariableOp_2:value:07fruit_tn1/loop_body/Tensordot_1/transpose/perm:output:0*
T0*"
_output_shapes
:~
-fruit_tn1/loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
'fruit_tn1/loop_body/Tensordot_1/ReshapeReshape-fruit_tn1/loop_body/Tensordot_1/transpose:y:06fruit_tn1/loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:?
/fruit_tn1/loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
)fruit_tn1/loop_body/Tensordot_1/Reshape_1Reshape&fruit_tn1/loop_body/Tensordot:output:08fruit_tn1/loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?(?
&fruit_tn1/loop_body/Tensordot_1/MatMulMatMul0fruit_tn1/loop_body/Tensordot_1/Reshape:output:02fruit_tn1/loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	?(?
%fruit_tn1/loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"               ?
fruit_tn1/loop_body/Tensordot_1Reshape0fruit_tn1/loop_body/Tensordot_1/MatMul:product:0.fruit_tn1/loop_body/Tensordot_1/shape:output:0*
T0*+
_output_shapes
:?~
-fruit_tn1/loop_body/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
'fruit_tn1/loop_body/Tensordot_2/ReshapeReshape,fruit_tn1/loop_body/ReadVariableOp_1:value:06fruit_tn1/loop_body/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes
:	?2?
.fruit_tn1/loop_body/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
)fruit_tn1/loop_body/Tensordot_2/transpose	Transpose(fruit_tn1/loop_body/Tensordot_1:output:07fruit_tn1/loop_body/Tensordot_2/transpose/perm:output:0*
T0*+
_output_shapes
:??
/fruit_tn1/loop_body/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
)fruit_tn1/loop_body/Tensordot_2/Reshape_1Reshape-fruit_tn1/loop_body/Tensordot_2/transpose:y:08fruit_tn1/loop_body/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2?
&fruit_tn1/loop_body/Tensordot_2/MatMulMatMul0fruit_tn1/loop_body/Tensordot_2/Reshape:output:02fruit_tn1/loop_body/Tensordot_2/Reshape_1:output:0*
T0*
_output_shapes

:z
%fruit_tn1/loop_body/Tensordot_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
fruit_tn1/loop_body/Tensordot_2Reshape0fruit_tn1/loop_body/Tensordot_2/MatMul:product:0.fruit_tn1/loop_body/Tensordot_2/shape:output:0*
T0*"
_output_shapes
:w
"fruit_tn1/loop_body/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
fruit_tn1/loop_body/transpose	Transpose(fruit_tn1/loop_body/Tensordot_2:output:0+fruit_tn1/loop_body/transpose/perm:output:0*
T0*"
_output_shapes
:?
&fruit_tn1/loop_body/add/ReadVariableOpReadVariableOp/fruit_tn1_loop_body_add_readvariableop_resource*"
_output_shapes
:*
dtype0?
fruit_tn1/loop_body/addAddV2!fruit_tn1/loop_body/transpose:y:0.fruit_tn1/loop_body/add/ReadVariableOp:value:0*
T0*"
_output_shapes
:f
fruit_tn1/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
fruit_tn1/pfor/ReshapeReshapefruit_tn1/Max:output:0%fruit_tn1/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:\
fruit_tn1/pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : \
fruit_tn1/pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
fruit_tn1/pfor/rangeRange#fruit_tn1/pfor/range/start:output:0fruit_tn1/Max:output:0#fruit_tn1/pfor/range/delta:output:0*#
_output_shapes
:?????????h
&fruit_tn1/loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : i
'fruit_tn1/loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
%fruit_tn1/loop_body/SelectV2/pfor/addAddV2/fruit_tn1/loop_body/SelectV2/pfor/Rank:output:00fruit_tn1/loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: j
(fruit_tn1/loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :j
(fruit_tn1/loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : k
)fruit_tn1/loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
'fruit_tn1/loop_body/SelectV2/pfor/add_1AddV21fruit_tn1/loop_body/SelectV2/pfor/Rank_2:output:02fruit_tn1/loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: ?
)fruit_tn1/loop_body/SelectV2/pfor/MaximumMaximum1fruit_tn1/loop_body/SelectV2/pfor/Rank_1:output:0)fruit_tn1/loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: ?
+fruit_tn1/loop_body/SelectV2/pfor/Maximum_1Maximum+fruit_tn1/loop_body/SelectV2/pfor/add_1:z:0-fruit_tn1/loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: t
'fruit_tn1/loop_body/SelectV2/pfor/ShapeShapefruit_tn1/pfor/range:output:0*
T0*
_output_shapes
:?
%fruit_tn1/loop_body/SelectV2/pfor/subSub/fruit_tn1/loop_body/SelectV2/pfor/Maximum_1:z:01fruit_tn1/loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: y
/fruit_tn1/loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
)fruit_tn1/loop_body/SelectV2/pfor/ReshapeReshape)fruit_tn1/loop_body/SelectV2/pfor/sub:z:08fruit_tn1/loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:v
,fruit_tn1/loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:?
&fruit_tn1/loop_body/SelectV2/pfor/TileTile5fruit_tn1/loop_body/SelectV2/pfor/Tile/input:output:02fruit_tn1/loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: 
5fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/fruit_tn1/loop_body/SelectV2/pfor/strided_sliceStridedSlice0fruit_tn1/loop_body/SelectV2/pfor/Shape:output:0>fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack:output:0@fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_1:output:0@fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
7fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
9fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1StridedSlice0fruit_tn1/loop_body/SelectV2/pfor/Shape:output:0@fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stack:output:0Bfruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:0Bfruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_masko
-fruit_tn1/loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(fruit_tn1/loop_body/SelectV2/pfor/concatConcatV28fruit_tn1/loop_body/SelectV2/pfor/strided_slice:output:0/fruit_tn1/loop_body/SelectV2/pfor/Tile:output:0:fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1:output:06fruit_tn1/loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
+fruit_tn1/loop_body/SelectV2/pfor/Reshape_1Reshapefruit_tn1/pfor/range:output:01fruit_tn1/loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:??????????
*fruit_tn1/loop_body/SelectV2/pfor/SelectV2SelectV2fruit_tn1/loop_body/Greater:z:04fruit_tn1/loop_body/SelectV2/pfor/Reshape_1:output:0'fruit_tn1/loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:?????????q
/fruit_tn1/loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*fruit_tn1/loop_body/GatherV2/pfor/GatherV2GatherV2 max_pooling2d_1/MaxPool:output:03fruit_tn1/loop_body/SelectV2/pfor/SelectV2:output:08fruit_tn1/loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*0
_output_shapes
:??????????t
2fruit_tn1/loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
0fruit_tn1/loop_body/Tensordot/transpose/pfor/addAddV25fruit_tn1/loop_body/Tensordot/transpose/perm:output:0;fruit_tn1/loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:?
<fruit_tn1/loop_body/Tensordot/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: z
8fruit_tn1/loop_body/Tensordot/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
3fruit_tn1/loop_body/Tensordot/transpose/pfor/concatConcatV2Efruit_tn1/loop_body/Tensordot/transpose/pfor/concat/values_0:output:04fruit_tn1/loop_body/Tensordot/transpose/pfor/add:z:0Afruit_tn1/loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
6fruit_tn1/loop_body/Tensordot/transpose/pfor/Transpose	Transpose3fruit_tn1/loop_body/GatherV2/pfor/GatherV2:output:0<fruit_tn1/loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*0
_output_shapes
:??????????x
6fruit_tn1/loop_body/Tensordot/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1fruit_tn1/loop_body/Tensordot/Reshape/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:04fruit_tn1/loop_body/Tensordot/Reshape/shape:output:0?fruit_tn1/loop_body/Tensordot/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
2fruit_tn1/loop_body/Tensordot/Reshape/pfor/ReshapeReshape:fruit_tn1/loop_body/Tensordot/transpose/pfor/Transpose:y:0:fruit_tn1/loop_body/Tensordot/Reshape/pfor/concat:output:0*
T0*,
_output_shapes
:???????????
/fruit_tn1/loop_body/Tensordot/MatMul/pfor/ShapeShape;fruit_tn1/loop_body/Tensordot/Reshape/pfor/Reshape:output:0*
T0*
_output_shapes
:{
9fruit_tn1/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
/fruit_tn1/loop_body/Tensordot/MatMul/pfor/splitSplitBfruit_tn1/loop_body/Tensordot/MatMul/pfor/split/split_dim:output:08fruit_tn1/loop_body/Tensordot/MatMul/pfor/Shape:output:0*
T0*&
_output_shapes
:::*
	num_splitz
7fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB |
9fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
1fruit_tn1/loop_body/Tensordot/MatMul/pfor/ReshapeReshape8fruit_tn1/loop_body/Tensordot/MatMul/pfor/split:output:0Bfruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1:output:0*
T0*
_output_shapes
: |
9fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ~
;fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
3fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1Reshape8fruit_tn1/loop_body/Tensordot/MatMul/pfor/split:output:1Dfruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: |
9fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ~
;fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
3fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2Reshape8fruit_tn1/loop_body/Tensordot/MatMul/pfor/split:output:2Dfruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ?
-fruit_tn1/loop_body/Tensordot/MatMul/pfor/mulMul:fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape:output:0<fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: ?
9fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack1fruit_tn1/loop_body/Tensordot/MatMul/pfor/mul:z:0<fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:?
3fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape;fruit_tn1/loop_body/Tensordot/Reshape/pfor/Reshape:output:0Bfruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:???????????????????
0fruit_tn1/loop_body/Tensordot/MatMul/pfor/MatMulMatMul<fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3:output:00fruit_tn1/loop_body/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:??????????
;fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
??????????
9fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePack:fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape:output:0<fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0Dfruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:?
3fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4Reshape:fruit_tn1/loop_body/Tensordot/MatMul/pfor/MatMul:product:0Bfruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*,
_output_shapes
:??????????p
.fruit_tn1/loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)fruit_tn1/loop_body/Tensordot/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:0,fruit_tn1/loop_body/Tensordot/shape:output:07fruit_tn1/loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
*fruit_tn1/loop_body/Tensordot/pfor/ReshapeReshape<fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4:output:02fruit_tn1/loop_body/Tensordot/pfor/concat:output:0*
T0*4
_output_shapes"
 :??????????|
:fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
5fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:08fruit_tn1/loop_body/Tensordot_1/Reshape_1/shape:output:0Cfruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
6fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape3fruit_tn1/loop_body/Tensordot/pfor/Reshape:output:0>fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:??????????(?
1fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/ShapeShape?fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:?
?fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSlice:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Hfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSlice:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/MinimumMinimumBfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0Dfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: u
3fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :?
1fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/EqualEqual7fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Minimum:z:0<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: ?
4fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          ?
4fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          ?
2fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/SelectSelect5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Equal:z:0=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/t:output:0=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:?
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSlice:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSlice:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSlice:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/stackPackDfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0Dfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0Dfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose	Transpose?fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'????????????????????????????
3fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/ReshapeReshape9fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose:y:0:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:??????????(?
3fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape_1Shape<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:}
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/splitSplitDfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split/split_dim:output:0<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split~
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1Reshape:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split:output:0Ffruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2Reshape:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split:output:1Ffruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3Reshape:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split:output:2Ffruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ?
/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/mulMul>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ?
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePack>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:03fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:?
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4Reshape<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0Dfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:???????????????????
2fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul0fruit_tn1/loop_body/Tensordot_1/Reshape:output:0>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:??????????
=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
??????????
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePackFfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:0>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:?
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5Reshape<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0Dfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'????????????????????????????
<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
7fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1	Transpose>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0Efruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????(r
0fruit_tn1/loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+fruit_tn1/loop_body/Tensordot_1/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:0.fruit_tn1/loop_body/Tensordot_1/shape:output:09fruit_tn1/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,fruit_tn1/loop_body/Tensordot_1/pfor/ReshapeReshape;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:04fruit_tn1/loop_body/Tensordot_1/pfor/concat:output:0*
T0*8
_output_shapes&
$:"??????????v
4fruit_tn1/loop_body/Tensordot_2/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
2fruit_tn1/loop_body/Tensordot_2/transpose/pfor/addAddV27fruit_tn1/loop_body/Tensordot_2/transpose/perm:output:0=fruit_tn1/loop_body/Tensordot_2/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:?
>fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: |
:fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
5fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concatConcatV2Gfruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/values_0:output:06fruit_tn1/loop_body/Tensordot_2/transpose/pfor/add:z:0Cfruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
8fruit_tn1/loop_body/Tensordot_2/transpose/pfor/Transpose	Transpose5fruit_tn1/loop_body/Tensordot_1/pfor/Reshape:output:0>fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat:output:0*
T0*8
_output_shapes&
$:"??????????|
:fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
5fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:08fruit_tn1/loop_body/Tensordot_2/Reshape_1/shape:output:0Cfruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
6fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/ReshapeReshape<fruit_tn1/loop_body/Tensordot_2/transpose/pfor/Transpose:y:0>fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:??????????2?
1fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/ShapeShape?fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:?
?fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_sliceStridedSlice:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Hfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1StridedSlice:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/MinimumMinimumBfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice:output:0Dfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: u
3fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :?
1fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/EqualEqual7fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Minimum:z:0<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: ?
4fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          ?
4fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          ?
2fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/SelectSelect5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Equal:z:0=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/t:output:0=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:?
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2StridedSlice:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3StridedSlice:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4StridedSlice:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/stackPackDfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2:output:0Dfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3:output:0Dfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose	Transpose?fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'????????????????????????????
3fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/ReshapeReshape9fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose:y:0:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:?2??????????
3fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape_1Shape<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:}
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/splitSplitDfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split/split_dim:output:0<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split~
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1Reshape:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split:output:0Ffruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2Reshape:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split:output:1Ffruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3Reshape:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split:output:2Ffruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ?
/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/mulMul>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ?
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shapePack>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1:output:03fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:?
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4Reshape<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0Dfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:???????????????????
2fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/MatMulMatMul0fruit_tn1/loop_body/Tensordot_2/Reshape:output:0>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:??????????
=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
??????????
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shapePackFfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0:output:0>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:?
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5Reshape<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/MatMul:product:0Dfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'????????????????????????????
<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
7fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1	Transpose>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5:output:0Efruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????r
0fruit_tn1/loop_body/Tensordot_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+fruit_tn1/loop_body/Tensordot_2/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:0.fruit_tn1/loop_body/Tensordot_2/shape:output:09fruit_tn1/loop_body/Tensordot_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,fruit_tn1/loop_body/Tensordot_2/pfor/ReshapeReshape;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1:y:04fruit_tn1/loop_body/Tensordot_2/pfor/concat:output:0*
T0*/
_output_shapes
:?????????j
(fruit_tn1/loop_body/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
&fruit_tn1/loop_body/transpose/pfor/addAddV2+fruit_tn1/loop_body/transpose/perm:output:01fruit_tn1/loop_body/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:|
2fruit_tn1/loop_body/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: p
.fruit_tn1/loop_body/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)fruit_tn1/loop_body/transpose/pfor/concatConcatV2;fruit_tn1/loop_body/transpose/pfor/concat/values_0:output:0*fruit_tn1/loop_body/transpose/pfor/add:z:07fruit_tn1/loop_body/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,fruit_tn1/loop_body/transpose/pfor/Transpose	Transpose5fruit_tn1/loop_body/Tensordot_2/pfor/Reshape:output:02fruit_tn1/loop_body/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:?????????c
!fruit_tn1/loop_body/add/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :e
#fruit_tn1/loop_body/add/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :d
"fruit_tn1/loop_body/add/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
 fruit_tn1/loop_body/add/pfor/addAddV2,fruit_tn1/loop_body/add/pfor/Rank_1:output:0+fruit_tn1/loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: ?
$fruit_tn1/loop_body/add/pfor/MaximumMaximum$fruit_tn1/loop_body/add/pfor/add:z:0*fruit_tn1/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: ?
"fruit_tn1/loop_body/add/pfor/ShapeShape0fruit_tn1/loop_body/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:?
 fruit_tn1/loop_body/add/pfor/subSub(fruit_tn1/loop_body/add/pfor/Maximum:z:0*fruit_tn1/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: t
*fruit_tn1/loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
$fruit_tn1/loop_body/add/pfor/ReshapeReshape$fruit_tn1/loop_body/add/pfor/sub:z:03fruit_tn1/loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:q
'fruit_tn1/loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:?
!fruit_tn1/loop_body/add/pfor/TileTile0fruit_tn1/loop_body/add/pfor/Tile/input:output:0-fruit_tn1/loop_body/add/pfor/Reshape:output:0*
T0*
_output_shapes
: z
0fruit_tn1/loop_body/add/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2fruit_tn1/loop_body/add/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2fruit_tn1/loop_body/add/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*fruit_tn1/loop_body/add/pfor/strided_sliceStridedSlice+fruit_tn1/loop_body/add/pfor/Shape:output:09fruit_tn1/loop_body/add/pfor/strided_slice/stack:output:0;fruit_tn1/loop_body/add/pfor/strided_slice/stack_1:output:0;fruit_tn1/loop_body/add/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
2fruit_tn1/loop_body/add/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:~
4fruit_tn1/loop_body/add/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ~
4fruit_tn1/loop_body/add/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,fruit_tn1/loop_body/add/pfor/strided_slice_1StridedSlice+fruit_tn1/loop_body/add/pfor/Shape:output:0;fruit_tn1/loop_body/add/pfor/strided_slice_1/stack:output:0=fruit_tn1/loop_body/add/pfor/strided_slice_1/stack_1:output:0=fruit_tn1/loop_body/add/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskj
(fruit_tn1/loop_body/add/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
#fruit_tn1/loop_body/add/pfor/concatConcatV23fruit_tn1/loop_body/add/pfor/strided_slice:output:0*fruit_tn1/loop_body/add/pfor/Tile:output:05fruit_tn1/loop_body/add/pfor/strided_slice_1:output:01fruit_tn1/loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
&fruit_tn1/loop_body/add/pfor/Reshape_1Reshape0fruit_tn1/loop_body/transpose/pfor/Transpose:y:0,fruit_tn1/loop_body/add/pfor/concat:output:0*
T0*/
_output_shapes
:??????????
"fruit_tn1/loop_body/add/pfor/AddV2AddV2/fruit_tn1/loop_body/add/pfor/Reshape_1:output:0.fruit_tn1/loop_body/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????h
fruit_tn1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
fruit_tn1/ReshapeReshape&fruit_tn1/loop_body/add/pfor/AddV2:z:0 fruit_tn1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????@d
fruit_tn1/ReluRelufruit_tn1/Reshape:output:0*
T0*'
_output_shapes
:?????????@~
$fruit_tn1/ActivityRegularizer/SquareSquarefruit_tn1/Relu:activations:0*
T0*'
_output_shapes
:?????????@t
#fruit_tn1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
!fruit_tn1/ActivityRegularizer/SumSum(fruit_tn1/ActivityRegularizer/Square:y:0,fruit_tn1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: h
#fruit_tn1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
!fruit_tn1/ActivityRegularizer/mulMul,fruit_tn1/ActivityRegularizer/mul/x:output:0*fruit_tn1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: o
#fruit_tn1/ActivityRegularizer/ShapeShapefruit_tn1/Relu:activations:0*
T0*
_output_shapes
:{
1fruit_tn1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3fruit_tn1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3fruit_tn1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+fruit_tn1/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn1/ActivityRegularizer/Shape:output:0:fruit_tn1/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"fruit_tn1/ActivityRegularizer/CastCast4fruit_tn1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%fruit_tn1/ActivityRegularizer/truedivRealDiv%fruit_tn1/ActivityRegularizer/mul:z:0&fruit_tn1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: [
dropout1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
dropout1/dropout/MulMulfruit_tn1/Relu:activations:0dropout1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@b
dropout1/dropout/ShapeShapefruit_tn1/Relu:activations:0*
T0*
_output_shapes
:?
-dropout1/dropout/random_uniform/RandomUniformRandomUniformdropout1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0d
dropout1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout1/dropout/GreaterEqualGreaterEqual6dropout1/dropout/random_uniform/RandomUniform:output:0(dropout1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
dropout1/dropout/CastCast!dropout1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
dropout1/dropout/Mul_1Muldropout1/dropout/Mul:z:0dropout1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@Y
fruit_tn2/ShapeShapedropout1/dropout/Mul_1:z:0*
T0*
_output_shapes
:g
fruit_tn2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
fruit_tn2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
fruit_tn2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
fruit_tn2/strided_sliceStridedSlicefruit_tn2/Shape:output:0&fruit_tn2/strided_slice/stack:output:0(fruit_tn2/strided_slice/stack_1:output:0(fruit_tn2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
fruit_tn2/Rank/packedPack fruit_tn2/strided_slice:output:0*
N*
T0*
_output_shapes
:P
fruit_tn2/RankConst*
_output_shapes
: *
dtype0*
value	B :W
fruit_tn2/range/startConst*
_output_shapes
: *
dtype0*
value	B : W
fruit_tn2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
fruit_tn2/rangeRangefruit_tn2/range/start:output:0fruit_tn2/Rank:output:0fruit_tn2/range/delta:output:0*
_output_shapes
:k
fruit_tn2/Max/inputPack fruit_tn2/strided_slice:output:0*
N*
T0*
_output_shapes
:m
fruit_tn2/MaxMaxfruit_tn2/Max/input:output:0fruit_tn2/range:output:0*
T0*
_output_shapes
: r
0fruit_tn2/loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : ?
*fruit_tn2/loop_body/PlaceholderWithDefaultPlaceholderWithDefault9fruit_tn2/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: c
fruit_tn2/loop_body/ShapeShapedropout1/dropout/Mul_1:z:0*
T0*
_output_shapes
:q
'fruit_tn2/loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)fruit_tn2/loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)fruit_tn2/loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!fruit_tn2/loop_body/strided_sliceStridedSlice"fruit_tn2/loop_body/Shape:output:00fruit_tn2/loop_body/strided_slice/stack:output:02fruit_tn2/loop_body/strided_slice/stack_1:output:02fruit_tn2/loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
fruit_tn2/loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :?
fruit_tn2/loop_body/GreaterGreater*fruit_tn2/loop_body/strided_slice:output:0&fruit_tn2/loop_body/Greater/y:output:0*
T0*
_output_shapes
: `
fruit_tn2/loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : ?
fruit_tn2/loop_body/SelectV2SelectV2fruit_tn2/loop_body/Greater:z:03fruit_tn2/loop_body/PlaceholderWithDefault:output:0'fruit_tn2/loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: c
!fruit_tn2/loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
fruit_tn2/loop_body/GatherV2GatherV2dropout1/dropout/Mul_1:z:0%fruit_tn2/loop_body/SelectV2:output:0*fruit_tn2/loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:@v
!fruit_tn2/loop_body/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
fruit_tn2/loop_body/ReshapeReshape%fruit_tn2/loop_body/GatherV2:output:0*fruit_tn2/loop_body/Reshape/shape:output:0*
T0*"
_output_shapes
:?
"fruit_tn2/loop_body/ReadVariableOpReadVariableOp+fruit_tn2_loop_body_readvariableop_resource*
_output_shapes

:*
dtype0?
$fruit_tn2/loop_body/ReadVariableOp_1ReadVariableOp-fruit_tn2_loop_body_readvariableop_1_resource*'
_output_shapes
:?*
dtype0?
$fruit_tn2/loop_body/ReadVariableOp_2ReadVariableOp-fruit_tn2_loop_body_readvariableop_2_resource*
_output_shapes

:*
dtype0?
,fruit_tn2/loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
'fruit_tn2/loop_body/Tensordot/transpose	Transpose$fruit_tn2/loop_body/Reshape:output:05fruit_tn2/loop_body/Tensordot/transpose/perm:output:0*
T0*"
_output_shapes
:|
+fruit_tn2/loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
%fruit_tn2/loop_body/Tensordot/ReshapeReshape+fruit_tn2/loop_body/Tensordot/transpose:y:04fruit_tn2/loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:?
$fruit_tn2/loop_body/Tensordot/MatMulMatMul.fruit_tn2/loop_body/Tensordot/Reshape:output:0*fruit_tn2/loop_body/ReadVariableOp:value:0*
T0*
_output_shapes

:x
#fruit_tn2/loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
fruit_tn2/loop_body/TensordotReshape.fruit_tn2/loop_body/Tensordot/MatMul:product:0,fruit_tn2/loop_body/Tensordot/shape:output:0*
T0*"
_output_shapes
:?
.fruit_tn2/loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
)fruit_tn2/loop_body/Tensordot_1/transpose	Transpose,fruit_tn2/loop_body/ReadVariableOp_1:value:07fruit_tn2/loop_body/Tensordot_1/transpose/perm:output:0*
T0*'
_output_shapes
:?~
-fruit_tn2/loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"     ?
'fruit_tn2/loop_body/Tensordot_1/ReshapeReshape-fruit_tn2/loop_body/Tensordot_1/transpose:y:06fruit_tn2/loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes
:	??
0fruit_tn2/loop_body/Tensordot_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
+fruit_tn2/loop_body/Tensordot_1/transpose_1	Transpose&fruit_tn2/loop_body/Tensordot:output:09fruit_tn2/loop_body/Tensordot_1/transpose_1/perm:output:0*
T0*"
_output_shapes
:?
/fruit_tn2/loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
)fruit_tn2/loop_body/Tensordot_1/Reshape_1Reshape/fruit_tn2/loop_body/Tensordot_1/transpose_1:y:08fruit_tn2/loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:?
&fruit_tn2/loop_body/Tensordot_1/MatMulMatMul0fruit_tn2/loop_body/Tensordot_1/Reshape:output:02fruit_tn2/loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	?z
%fruit_tn2/loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"?         ?
fruit_tn2/loop_body/Tensordot_1Reshape0fruit_tn2/loop_body/Tensordot_1/MatMul:product:0.fruit_tn2/loop_body/Tensordot_1/shape:output:0*
T0*#
_output_shapes
:?~
-fruit_tn2/loop_body/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
'fruit_tn2/loop_body/Tensordot_2/ReshapeReshape,fruit_tn2/loop_body/ReadVariableOp_2:value:06fruit_tn2/loop_body/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes

:?
.fruit_tn2/loop_body/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
)fruit_tn2/loop_body/Tensordot_2/transpose	Transpose(fruit_tn2/loop_body/Tensordot_1:output:07fruit_tn2/loop_body/Tensordot_2/transpose/perm:output:0*
T0*#
_output_shapes
:??
/fruit_tn2/loop_body/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   ?
)fruit_tn2/loop_body/Tensordot_2/Reshape_1Reshape-fruit_tn2/loop_body/Tensordot_2/transpose:y:08fruit_tn2/loop_body/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	??
&fruit_tn2/loop_body/Tensordot_2/MatMulMatMul0fruit_tn2/loop_body/Tensordot_2/Reshape:output:02fruit_tn2/loop_body/Tensordot_2/Reshape_1:output:0*
T0*
_output_shapes
:	?p
%fruit_tn2/loop_body/Tensordot_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:??
fruit_tn2/loop_body/Tensordot_2Reshape0fruit_tn2/loop_body/Tensordot_2/MatMul:product:0.fruit_tn2/loop_body/Tensordot_2/shape:output:0*
T0*
_output_shapes	
:??
&fruit_tn2/loop_body/add/ReadVariableOpReadVariableOp/fruit_tn2_loop_body_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
fruit_tn2/loop_body/addAddV2(fruit_tn2/loop_body/Tensordot_2:output:0.fruit_tn2/loop_body/add/ReadVariableOp:value:0*
T0*
_output_shapes	
:?f
fruit_tn2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
fruit_tn2/pfor/ReshapeReshapefruit_tn2/Max:output:0%fruit_tn2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:\
fruit_tn2/pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : \
fruit_tn2/pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
fruit_tn2/pfor/rangeRange#fruit_tn2/pfor/range/start:output:0fruit_tn2/Max:output:0#fruit_tn2/pfor/range/delta:output:0*#
_output_shapes
:?????????h
&fruit_tn2/loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : i
'fruit_tn2/loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
%fruit_tn2/loop_body/SelectV2/pfor/addAddV2/fruit_tn2/loop_body/SelectV2/pfor/Rank:output:00fruit_tn2/loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: j
(fruit_tn2/loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :j
(fruit_tn2/loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : k
)fruit_tn2/loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
'fruit_tn2/loop_body/SelectV2/pfor/add_1AddV21fruit_tn2/loop_body/SelectV2/pfor/Rank_2:output:02fruit_tn2/loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: ?
)fruit_tn2/loop_body/SelectV2/pfor/MaximumMaximum1fruit_tn2/loop_body/SelectV2/pfor/Rank_1:output:0)fruit_tn2/loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: ?
+fruit_tn2/loop_body/SelectV2/pfor/Maximum_1Maximum+fruit_tn2/loop_body/SelectV2/pfor/add_1:z:0-fruit_tn2/loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: t
'fruit_tn2/loop_body/SelectV2/pfor/ShapeShapefruit_tn2/pfor/range:output:0*
T0*
_output_shapes
:?
%fruit_tn2/loop_body/SelectV2/pfor/subSub/fruit_tn2/loop_body/SelectV2/pfor/Maximum_1:z:01fruit_tn2/loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: y
/fruit_tn2/loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
)fruit_tn2/loop_body/SelectV2/pfor/ReshapeReshape)fruit_tn2/loop_body/SelectV2/pfor/sub:z:08fruit_tn2/loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:v
,fruit_tn2/loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:?
&fruit_tn2/loop_body/SelectV2/pfor/TileTile5fruit_tn2/loop_body/SelectV2/pfor/Tile/input:output:02fruit_tn2/loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: 
5fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/fruit_tn2/loop_body/SelectV2/pfor/strided_sliceStridedSlice0fruit_tn2/loop_body/SelectV2/pfor/Shape:output:0>fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack:output:0@fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_1:output:0@fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
7fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
9fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1StridedSlice0fruit_tn2/loop_body/SelectV2/pfor/Shape:output:0@fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stack:output:0Bfruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:0Bfruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_masko
-fruit_tn2/loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(fruit_tn2/loop_body/SelectV2/pfor/concatConcatV28fruit_tn2/loop_body/SelectV2/pfor/strided_slice:output:0/fruit_tn2/loop_body/SelectV2/pfor/Tile:output:0:fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1:output:06fruit_tn2/loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
+fruit_tn2/loop_body/SelectV2/pfor/Reshape_1Reshapefruit_tn2/pfor/range:output:01fruit_tn2/loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:??????????
*fruit_tn2/loop_body/SelectV2/pfor/SelectV2SelectV2fruit_tn2/loop_body/Greater:z:04fruit_tn2/loop_body/SelectV2/pfor/Reshape_1:output:0'fruit_tn2/loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:?????????q
/fruit_tn2/loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*fruit_tn2/loop_body/GatherV2/pfor/GatherV2GatherV2dropout1/dropout/Mul_1:z:03fruit_tn2/loop_body/SelectV2/pfor/SelectV2:output:08fruit_tn2/loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:?????????@n
,fruit_tn2/loop_body/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'fruit_tn2/loop_body/Reshape/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:0*fruit_tn2/loop_body/Reshape/shape:output:05fruit_tn2/loop_body/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
(fruit_tn2/loop_body/Reshape/pfor/ReshapeReshape3fruit_tn2/loop_body/GatherV2/pfor/GatherV2:output:00fruit_tn2/loop_body/Reshape/pfor/concat:output:0*
T0*/
_output_shapes
:?????????t
2fruit_tn2/loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
0fruit_tn2/loop_body/Tensordot/transpose/pfor/addAddV25fruit_tn2/loop_body/Tensordot/transpose/perm:output:0;fruit_tn2/loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:?
<fruit_tn2/loop_body/Tensordot/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: z
8fruit_tn2/loop_body/Tensordot/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
3fruit_tn2/loop_body/Tensordot/transpose/pfor/concatConcatV2Efruit_tn2/loop_body/Tensordot/transpose/pfor/concat/values_0:output:04fruit_tn2/loop_body/Tensordot/transpose/pfor/add:z:0Afruit_tn2/loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
6fruit_tn2/loop_body/Tensordot/transpose/pfor/Transpose	Transpose1fruit_tn2/loop_body/Reshape/pfor/Reshape:output:0<fruit_tn2/loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:?????????x
6fruit_tn2/loop_body/Tensordot/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1fruit_tn2/loop_body/Tensordot/Reshape/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:04fruit_tn2/loop_body/Tensordot/Reshape/shape:output:0?fruit_tn2/loop_body/Tensordot/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
2fruit_tn2/loop_body/Tensordot/Reshape/pfor/ReshapeReshape:fruit_tn2/loop_body/Tensordot/transpose/pfor/Transpose:y:0:fruit_tn2/loop_body/Tensordot/Reshape/pfor/concat:output:0*
T0*+
_output_shapes
:??????????
/fruit_tn2/loop_body/Tensordot/MatMul/pfor/ShapeShape;fruit_tn2/loop_body/Tensordot/Reshape/pfor/Reshape:output:0*
T0*
_output_shapes
:{
9fruit_tn2/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
/fruit_tn2/loop_body/Tensordot/MatMul/pfor/splitSplitBfruit_tn2/loop_body/Tensordot/MatMul/pfor/split/split_dim:output:08fruit_tn2/loop_body/Tensordot/MatMul/pfor/Shape:output:0*
T0*&
_output_shapes
:::*
	num_splitz
7fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB |
9fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
1fruit_tn2/loop_body/Tensordot/MatMul/pfor/ReshapeReshape8fruit_tn2/loop_body/Tensordot/MatMul/pfor/split:output:0Bfruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1:output:0*
T0*
_output_shapes
: |
9fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ~
;fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
3fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1Reshape8fruit_tn2/loop_body/Tensordot/MatMul/pfor/split:output:1Dfruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: |
9fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ~
;fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
3fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2Reshape8fruit_tn2/loop_body/Tensordot/MatMul/pfor/split:output:2Dfruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ?
-fruit_tn2/loop_body/Tensordot/MatMul/pfor/mulMul:fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape:output:0<fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: ?
9fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack1fruit_tn2/loop_body/Tensordot/MatMul/pfor/mul:z:0<fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:?
3fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape;fruit_tn2/loop_body/Tensordot/Reshape/pfor/Reshape:output:0Bfruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:???????????????????
0fruit_tn2/loop_body/Tensordot/MatMul/pfor/MatMulMatMul<fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0*fruit_tn2/loop_body/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
;fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
??????????
9fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePack:fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape:output:0<fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0Dfruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:?
3fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4Reshape:fruit_tn2/loop_body/Tensordot/MatMul/pfor/MatMul:product:0Bfruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*+
_output_shapes
:?????????p
.fruit_tn2/loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)fruit_tn2/loop_body/Tensordot/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:0,fruit_tn2/loop_body/Tensordot/shape:output:07fruit_tn2/loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
*fruit_tn2/loop_body/Tensordot/pfor/ReshapeReshape<fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4:output:02fruit_tn2/loop_body/Tensordot/pfor/concat:output:0*
T0*/
_output_shapes
:?????????x
6fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
4fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/addAddV29fruit_tn2/loop_body/Tensordot_1/transpose_1/perm:output:0?fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/add/y:output:0*
T0*
_output_shapes
:?
@fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: ~
<fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
7fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concatConcatV2Ifruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/values_0:output:08fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/add:z:0Efruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
:fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/Transpose	Transpose3fruit_tn2/loop_body/Tensordot/pfor/Reshape:output:0@fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat:output:0*
T0*/
_output_shapes
:?????????|
:fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
5fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:08fruit_tn2/loop_body/Tensordot_1/Reshape_1/shape:output:0Cfruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
6fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape>fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/Transpose:y:0>fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*+
_output_shapes
:??????????
1fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/ShapeShape?fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:?
?fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSlice:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Hfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSlice:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/MinimumMinimumBfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0Dfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: u
3fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :?
1fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/EqualEqual7fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Minimum:z:0<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: ?
4fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          ?
4fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          ?
2fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/SelectSelect5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Equal:z:0=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/t:output:0=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:?
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSlice:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSlice:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSlice:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/stackPackDfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0Dfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0Dfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose	Transpose?fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'????????????????????????????
3fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/ReshapeReshape9fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose:y:0:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*+
_output_shapes
:??????????
3fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape_1Shape<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:}
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/splitSplitDfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split/split_dim:output:0<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split~
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1Reshape:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split:output:0Ffruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2Reshape:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split:output:1Ffruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3Reshape:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split:output:2Ffruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ?
/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/mulMul>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ?
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePack>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:03fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:?
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4Reshape<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0Dfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:???????????????????
2fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul0fruit_tn2/loop_body/Tensordot_1/Reshape:output:0>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*(
_output_shapes
:???????????
=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
??????????
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePackFfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:0>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:?
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5Reshape<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0Dfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'????????????????????????????
<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
7fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1	Transpose>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0Efruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????r
0fruit_tn2/loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+fruit_tn2/loop_body/Tensordot_1/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:0.fruit_tn2/loop_body/Tensordot_1/shape:output:09fruit_tn2/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,fruit_tn2/loop_body/Tensordot_1/pfor/ReshapeReshape;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:04fruit_tn2/loop_body/Tensordot_1/pfor/concat:output:0*
T0*0
_output_shapes
:??????????v
4fruit_tn2/loop_body/Tensordot_2/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
2fruit_tn2/loop_body/Tensordot_2/transpose/pfor/addAddV27fruit_tn2/loop_body/Tensordot_2/transpose/perm:output:0=fruit_tn2/loop_body/Tensordot_2/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:?
>fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: |
:fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
5fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concatConcatV2Gfruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/values_0:output:06fruit_tn2/loop_body/Tensordot_2/transpose/pfor/add:z:0Cfruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
8fruit_tn2/loop_body/Tensordot_2/transpose/pfor/Transpose	Transpose5fruit_tn2/loop_body/Tensordot_1/pfor/Reshape:output:0>fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat:output:0*
T0*0
_output_shapes
:??????????|
:fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
5fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:08fruit_tn2/loop_body/Tensordot_2/Reshape_1/shape:output:0Cfruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
6fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/ReshapeReshape<fruit_tn2/loop_body/Tensordot_2/transpose/pfor/Transpose:y:0>fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:???????????
1fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/ShapeShape?fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:?
?fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_sliceStridedSlice:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Hfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1StridedSlice:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/MinimumMinimumBfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice:output:0Dfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: u
3fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :?
1fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/EqualEqual7fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Minimum:z:0<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: ?
4fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          ?
4fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          ?
2fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/SelectSelect5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Equal:z:0=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/t:output:0=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:?
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2StridedSlice:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3StridedSlice:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4StridedSlice:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/stackPackDfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2:output:0Dfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3:output:0Dfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose	Transpose?fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'????????????????????????????
3fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/ReshapeReshape9fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose:y:0:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:???????????
3fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape_1Shape<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:}
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/splitSplitDfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split/split_dim:output:0<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split~
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1Reshape:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split:output:0Ffruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2Reshape:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split:output:1Ffruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3Reshape:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split:output:2Ffruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ?
/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/mulMul>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ?
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shapePack>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1:output:03fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:?
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4Reshape<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0Dfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:???????????????????
2fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/MatMulMatMul0fruit_tn2/loop_body/Tensordot_2/Reshape:output:0>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:??????????
=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
??????????
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shapePackFfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0:output:0>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:?
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5Reshape<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/MatMul:product:0Dfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'????????????????????????????
<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
7fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1	Transpose>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5:output:0Efruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????r
0fruit_tn2/loop_body/Tensordot_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+fruit_tn2/loop_body/Tensordot_2/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:0.fruit_tn2/loop_body/Tensordot_2/shape:output:09fruit_tn2/loop_body/Tensordot_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,fruit_tn2/loop_body/Tensordot_2/pfor/ReshapeReshape;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1:y:04fruit_tn2/loop_body/Tensordot_2/pfor/concat:output:0*
T0*(
_output_shapes
:??????????c
!fruit_tn2/loop_body/add/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :e
#fruit_tn2/loop_body/add/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :d
"fruit_tn2/loop_body/add/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
 fruit_tn2/loop_body/add/pfor/addAddV2,fruit_tn2/loop_body/add/pfor/Rank_1:output:0+fruit_tn2/loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: ?
$fruit_tn2/loop_body/add/pfor/MaximumMaximum$fruit_tn2/loop_body/add/pfor/add:z:0*fruit_tn2/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: ?
"fruit_tn2/loop_body/add/pfor/ShapeShape5fruit_tn2/loop_body/Tensordot_2/pfor/Reshape:output:0*
T0*
_output_shapes
:?
 fruit_tn2/loop_body/add/pfor/subSub(fruit_tn2/loop_body/add/pfor/Maximum:z:0*fruit_tn2/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: t
*fruit_tn2/loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
$fruit_tn2/loop_body/add/pfor/ReshapeReshape$fruit_tn2/loop_body/add/pfor/sub:z:03fruit_tn2/loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:q
'fruit_tn2/loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:?
!fruit_tn2/loop_body/add/pfor/TileTile0fruit_tn2/loop_body/add/pfor/Tile/input:output:0-fruit_tn2/loop_body/add/pfor/Reshape:output:0*
T0*
_output_shapes
: z
0fruit_tn2/loop_body/add/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2fruit_tn2/loop_body/add/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2fruit_tn2/loop_body/add/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*fruit_tn2/loop_body/add/pfor/strided_sliceStridedSlice+fruit_tn2/loop_body/add/pfor/Shape:output:09fruit_tn2/loop_body/add/pfor/strided_slice/stack:output:0;fruit_tn2/loop_body/add/pfor/strided_slice/stack_1:output:0;fruit_tn2/loop_body/add/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
2fruit_tn2/loop_body/add/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:~
4fruit_tn2/loop_body/add/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ~
4fruit_tn2/loop_body/add/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,fruit_tn2/loop_body/add/pfor/strided_slice_1StridedSlice+fruit_tn2/loop_body/add/pfor/Shape:output:0;fruit_tn2/loop_body/add/pfor/strided_slice_1/stack:output:0=fruit_tn2/loop_body/add/pfor/strided_slice_1/stack_1:output:0=fruit_tn2/loop_body/add/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskj
(fruit_tn2/loop_body/add/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
#fruit_tn2/loop_body/add/pfor/concatConcatV23fruit_tn2/loop_body/add/pfor/strided_slice:output:0*fruit_tn2/loop_body/add/pfor/Tile:output:05fruit_tn2/loop_body/add/pfor/strided_slice_1:output:01fruit_tn2/loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
&fruit_tn2/loop_body/add/pfor/Reshape_1Reshape5fruit_tn2/loop_body/Tensordot_2/pfor/Reshape:output:0,fruit_tn2/loop_body/add/pfor/concat:output:0*
T0*(
_output_shapes
:???????????
"fruit_tn2/loop_body/add/pfor/AddV2AddV2/fruit_tn2/loop_body/add/pfor/Reshape_1:output:0.fruit_tn2/loop_body/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????h
fruit_tn2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
fruit_tn2/ReshapeReshape&fruit_tn2/loop_body/add/pfor/AddV2:z:0 fruit_tn2/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????k
fruit_tn2/SoftmaxSoftmaxfruit_tn2/Reshape:output:0*
T0*(
_output_shapes
:??????????~
$fruit_tn2/ActivityRegularizer/SquareSquarefruit_tn2/Softmax:softmax:0*
T0*(
_output_shapes
:??????????t
#fruit_tn2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
!fruit_tn2/ActivityRegularizer/SumSum(fruit_tn2/ActivityRegularizer/Square:y:0,fruit_tn2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: h
#fruit_tn2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
!fruit_tn2/ActivityRegularizer/mulMul,fruit_tn2/ActivityRegularizer/mul/x:output:0*fruit_tn2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: n
#fruit_tn2/ActivityRegularizer/ShapeShapefruit_tn2/Softmax:softmax:0*
T0*
_output_shapes
:{
1fruit_tn2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3fruit_tn2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3fruit_tn2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+fruit_tn2/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn2/ActivityRegularizer/Shape:output:0:fruit_tn2/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"fruit_tn2/ActivityRegularizer/CastCast4fruit_tn2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%fruit_tn2/ActivityRegularizer/truedivRealDiv%fruit_tn2/ActivityRegularizer/mul:z:0&fruit_tn2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: k
IdentityIdentityfruit_tn2/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:??????????i

Identity_1Identity)conv2dmpo/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: k

Identity_2Identity+conv2dmpo_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_3Identity)fruit_tn1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_4Identity)fruit_tn2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^conv2dmpo/ReadVariableOp^conv2dmpo/ReadVariableOp_1#^conv2dmpo/Reshape_2/ReadVariableOp^conv2dmpo_1/ReadVariableOp^conv2dmpo_1/ReadVariableOp_1^conv2dmpo_1/ReadVariableOp_2^conv2dmpo_1/ReadVariableOp_3%^conv2dmpo_1/Reshape_2/ReadVariableOp#^fruit_tn1/loop_body/ReadVariableOp%^fruit_tn1/loop_body/ReadVariableOp_1%^fruit_tn1/loop_body/ReadVariableOp_2'^fruit_tn1/loop_body/add/ReadVariableOp#^fruit_tn2/loop_body/ReadVariableOp%^fruit_tn2/loop_body/ReadVariableOp_1%^fruit_tn2/loop_body/ReadVariableOp_2'^fruit_tn2/loop_body/add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????22: : : : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp24
conv2dmpo/ReadVariableOpconv2dmpo/ReadVariableOp28
conv2dmpo/ReadVariableOp_1conv2dmpo/ReadVariableOp_12H
"conv2dmpo/Reshape_2/ReadVariableOp"conv2dmpo/Reshape_2/ReadVariableOp28
conv2dmpo_1/ReadVariableOpconv2dmpo_1/ReadVariableOp2<
conv2dmpo_1/ReadVariableOp_1conv2dmpo_1/ReadVariableOp_12<
conv2dmpo_1/ReadVariableOp_2conv2dmpo_1/ReadVariableOp_22<
conv2dmpo_1/ReadVariableOp_3conv2dmpo_1/ReadVariableOp_32L
$conv2dmpo_1/Reshape_2/ReadVariableOp$conv2dmpo_1/Reshape_2/ReadVariableOp2H
"fruit_tn1/loop_body/ReadVariableOp"fruit_tn1/loop_body/ReadVariableOp2L
$fruit_tn1/loop_body/ReadVariableOp_1$fruit_tn1/loop_body/ReadVariableOp_12L
$fruit_tn1/loop_body/ReadVariableOp_2$fruit_tn1/loop_body/ReadVariableOp_22P
&fruit_tn1/loop_body/add/ReadVariableOp&fruit_tn1/loop_body/add/ReadVariableOp2H
"fruit_tn2/loop_body/ReadVariableOp"fruit_tn2/loop_body/ReadVariableOp2L
$fruit_tn2/loop_body/ReadVariableOp_1$fruit_tn2/loop_body/ReadVariableOp_12L
$fruit_tn2/loop_body/ReadVariableOp_2$fruit_tn2/loop_body/ReadVariableOp_22P
&fruit_tn2/loop_body/add/ReadVariableOp&fruit_tn2/loop_body/add/ReadVariableOp:W S
/
_output_shapes
:?????????22
 
_user_specified_nameinputs

?
H__inference_sequential_1_layer_call_and_return_conditional_losses_414698

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:;
!conv2dmpo_readvariableop_resource:=
#conv2dmpo_readvariableop_1_resource:9
+conv2dmpo_reshape_2_readvariableop_resource:=
#conv2dmpo_1_readvariableop_resource:?
%conv2dmpo_1_readvariableop_1_resource:?
%conv2dmpo_1_readvariableop_2_resource:?
%conv2dmpo_1_readvariableop_3_resource:<
-conv2dmpo_1_reshape_2_readvariableop_resource:	?A
+fruit_tn1_loop_body_readvariableop_resource:H
-fruit_tn1_loop_body_readvariableop_1_resource:?C
-fruit_tn1_loop_body_readvariableop_2_resource:E
/fruit_tn1_loop_body_add_readvariableop_resource:=
+fruit_tn2_loop_body_readvariableop_resource:H
-fruit_tn2_loop_body_readvariableop_1_resource:??
-fruit_tn2_loop_body_readvariableop_2_resource:>
/fruit_tn2_loop_body_add_readvariableop_resource:	?
identity

identity_1

identity_2

identity_3

identity_4??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2dmpo/ReadVariableOp?conv2dmpo/ReadVariableOp_1?"conv2dmpo/Reshape_2/ReadVariableOp?conv2dmpo_1/ReadVariableOp?conv2dmpo_1/ReadVariableOp_1?conv2dmpo_1/ReadVariableOp_2?conv2dmpo_1/ReadVariableOp_3?$conv2dmpo_1/Reshape_2/ReadVariableOp?"fruit_tn1/loop_body/ReadVariableOp?$fruit_tn1/loop_body/ReadVariableOp_1?$fruit_tn1/loop_body/ReadVariableOp_2?&fruit_tn1/loop_body/add/ReadVariableOp?"fruit_tn2/loop_body/ReadVariableOp?$fruit_tn2/loop_body/ReadVariableOp_1?$fruit_tn2/loop_body/ReadVariableOp_2?&fruit_tn2/loop_body/add/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????//*
paddingVALID*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????//?
conv2dmpo/ReadVariableOpReadVariableOp!conv2dmpo_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2dmpo/ReadVariableOp_1ReadVariableOp#conv2dmpo_readvariableop_1_resource*&
_output_shapes
:*
dtype0{
"conv2dmpo/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2dmpo/Tensordot/transpose	Transpose conv2dmpo/ReadVariableOp:value:0+conv2dmpo/Tensordot/transpose/perm:output:0*
T0*&
_output_shapes
:r
!conv2dmpo/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
conv2dmpo/Tensordot/ReshapeReshape!conv2dmpo/Tensordot/transpose:y:0*conv2dmpo/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:}
$conv2dmpo/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2dmpo/Tensordot/transpose_1	Transpose"conv2dmpo/ReadVariableOp_1:value:0-conv2dmpo/Tensordot/transpose_1/perm:output:0*
T0*&
_output_shapes
:t
#conv2dmpo/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
conv2dmpo/Tensordot/Reshape_1Reshape#conv2dmpo/Tensordot/transpose_1:y:0,conv2dmpo/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:?
conv2dmpo/Tensordot/MatMulMatMul$conv2dmpo/Tensordot/Reshape:output:0&conv2dmpo/Tensordot/Reshape_1:output:0*
T0*
_output_shapes

:z
conv2dmpo/Tensordot/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  ?
conv2dmpo/TensordotReshape$conv2dmpo/Tensordot/MatMul:product:0"conv2dmpo/Tensordot/shape:output:0*
T0*.
_output_shapes
:y
conv2dmpo/transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
conv2dmpo/transpose	Transposeconv2dmpo/Tensordot:output:0!conv2dmpo/transpose/perm:output:0*
T0*.
_output_shapes
:{
conv2dmpo/transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
conv2dmpo/transpose_1	Transposeconv2dmpo/transpose:y:0#conv2dmpo/transpose_1/perm:output:0*
T0*.
_output_shapes
:p
conv2dmpo/ShapeConst*
_output_shapes
:*
dtype0*-
value$B""                  g
conv2dmpo/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
conv2dmpo/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
conv2dmpo/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2dmpo/strided_sliceStridedSliceconv2dmpo/Shape:output:0&conv2dmpo/strided_slice/stack:output:0(conv2dmpo/strided_slice/stack_1:output:0(conv2dmpo/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskY
conv2dmpo/ConstConst*
_output_shapes
:*
dtype0*
valueB: s
conv2dmpo/ProdProd conv2dmpo/strided_slice:output:0conv2dmpo/Const:output:0*
T0*
_output_shapes
: i
conv2dmpo/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!conv2dmpo/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!conv2dmpo/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2dmpo/strided_slice_1StridedSliceconv2dmpo/Shape:output:0(conv2dmpo/strided_slice_1/stack:output:0*conv2dmpo/strided_slice_1/stack_1:output:0*conv2dmpo/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
conv2dmpo/concat/values_1Packconv2dmpo/Prod:output:0*
N*
T0*
_output_shapes
:`
conv2dmpo/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv2dmpo/concatConcatV2"conv2dmpo/strided_slice_1:output:0"conv2dmpo/concat/values_1:output:0conv2dmpo/concat/axis:output:0*
N*
T0*
_output_shapes
:?
conv2dmpo/ReshapeReshapeconv2dmpo/transpose_1:y:0conv2dmpo/concat:output:0*
T0**
_output_shapes
:w
conv2dmpo/transpose_2/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
conv2dmpo/transpose_2	Transposeconv2dmpo/Reshape:output:0#conv2dmpo/transpose_2/perm:output:0*
T0**
_output_shapes
:n
conv2dmpo/Shape_1Const*
_output_shapes
:*
dtype0*)
value B"               i
conv2dmpo/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:k
!conv2dmpo/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: k
!conv2dmpo/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2dmpo/strided_slice_2StridedSliceconv2dmpo/Shape_1:output:0(conv2dmpo/strided_slice_2/stack:output:0*conv2dmpo/strided_slice_2/stack_1:output:0*conv2dmpo/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask[
conv2dmpo/Const_1Const*
_output_shapes
:*
dtype0*
valueB: y
conv2dmpo/Prod_1Prod"conv2dmpo/strided_slice_2:output:0conv2dmpo/Const_1:output:0*
T0*
_output_shapes
: i
conv2dmpo/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!conv2dmpo/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!conv2dmpo/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2dmpo/strided_slice_3StridedSliceconv2dmpo/Shape_1:output:0(conv2dmpo/strided_slice_3/stack:output:0*conv2dmpo/strided_slice_3/stack_1:output:0*conv2dmpo/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskl
conv2dmpo/concat_1/values_1Packconv2dmpo/Prod_1:output:0*
N*
T0*
_output_shapes
:b
conv2dmpo/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv2dmpo/concat_1ConcatV2"conv2dmpo/strided_slice_3:output:0$conv2dmpo/concat_1/values_1:output:0 conv2dmpo/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
conv2dmpo/Reshape_1Reshapeconv2dmpo/transpose_2:y:0conv2dmpo/concat_1:output:0*
T0*&
_output_shapes
:?
conv2dmpo/Conv2DConv2Dconv2d/BiasAdd:output:0conv2dmpo/Reshape_1:output:0*
T0*/
_output_shapes
:?????????//*
paddingSAME*
strides
?
"conv2dmpo/Reshape_2/ReadVariableOpReadVariableOp+conv2dmpo_reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0j
conv2dmpo/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
conv2dmpo/Reshape_2Reshape*conv2dmpo/Reshape_2/ReadVariableOp:value:0"conv2dmpo/Reshape_2/shape:output:0*
T0*
_output_shapes

:?
conv2dmpo/addAddV2conv2dmpo/Conv2D:output:0conv2dmpo/Reshape_2:output:0*
T0*/
_output_shapes
:?????????//c
conv2dmpo/ReluReluconv2dmpo/add:z:0*
T0*/
_output_shapes
:?????????//?
$conv2dmpo/ActivityRegularizer/SquareSquareconv2dmpo/Relu:activations:0*
T0*/
_output_shapes
:?????????//|
#conv2dmpo/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2dmpo/ActivityRegularizer/SumSum(conv2dmpo/ActivityRegularizer/Square:y:0,conv2dmpo/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: h
#conv2dmpo/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
!conv2dmpo/ActivityRegularizer/mulMul,conv2dmpo/ActivityRegularizer/mul/x:output:0*conv2dmpo/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: o
#conv2dmpo/ActivityRegularizer/ShapeShapeconv2dmpo/Relu:activations:0*
T0*
_output_shapes
:{
1conv2dmpo/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3conv2dmpo/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3conv2dmpo/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+conv2dmpo/ActivityRegularizer/strided_sliceStridedSlice,conv2dmpo/ActivityRegularizer/Shape:output:0:conv2dmpo/ActivityRegularizer/strided_slice/stack:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_1:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"conv2dmpo/ActivityRegularizer/CastCast4conv2dmpo/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%conv2dmpo/ActivityRegularizer/truedivRealDiv%conv2dmpo/ActivityRegularizer/mul:z:0&conv2dmpo/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
max_pooling2d/MaxPoolMaxPoolconv2dmpo/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
?
conv2dmpo_1/ReadVariableOpReadVariableOp#conv2dmpo_1_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2dmpo_1/ReadVariableOp_1ReadVariableOp%conv2dmpo_1_readvariableop_1_resource*&
_output_shapes
:*
dtype0?
conv2dmpo_1/ReadVariableOp_2ReadVariableOp%conv2dmpo_1_readvariableop_2_resource*&
_output_shapes
:*
dtype0?
conv2dmpo_1/ReadVariableOp_3ReadVariableOp%conv2dmpo_1_readvariableop_3_resource*&
_output_shapes
:*
dtype0}
$conv2dmpo_1/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2dmpo_1/Tensordot/transpose	Transpose$conv2dmpo_1/ReadVariableOp_2:value:0-conv2dmpo_1/Tensordot/transpose/perm:output:0*
T0*&
_output_shapes
:t
#conv2dmpo_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@      ?
conv2dmpo_1/Tensordot/ReshapeReshape#conv2dmpo_1/Tensordot/transpose:y:0,conv2dmpo_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@
&conv2dmpo_1/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2dmpo_1/Tensordot/transpose_1	Transpose$conv2dmpo_1/ReadVariableOp_3:value:0/conv2dmpo_1/Tensordot/transpose_1/perm:output:0*
T0*&
_output_shapes
:v
%conv2dmpo_1/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
conv2dmpo_1/Tensordot/Reshape_1Reshape%conv2dmpo_1/Tensordot/transpose_1:y:0.conv2dmpo_1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:?
conv2dmpo_1/Tensordot/MatMulMatMul&conv2dmpo_1/Tensordot/Reshape:output:0(conv2dmpo_1/Tensordot/Reshape_1:output:0*
T0*
_output_shapes

:@|
conv2dmpo_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  ?
conv2dmpo_1/TensordotReshape&conv2dmpo_1/Tensordot/MatMul:product:0$conv2dmpo_1/Tensordot/shape:output:0*
T0*.
_output_shapes
:
&conv2dmpo_1/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2dmpo_1/Tensordot_1/transpose	Transpose$conv2dmpo_1/ReadVariableOp_1:value:0/conv2dmpo_1/Tensordot_1/transpose/perm:output:0*
T0*&
_output_shapes
:v
%conv2dmpo_1/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@      ?
conv2dmpo_1/Tensordot_1/ReshapeReshape%conv2dmpo_1/Tensordot_1/transpose:y:0.conv2dmpo_1/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:@?
(conv2dmpo_1/Tensordot_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
#conv2dmpo_1/Tensordot_1/transpose_1	Transpose"conv2dmpo_1/ReadVariableOp:value:01conv2dmpo_1/Tensordot_1/transpose_1/perm:output:0*
T0*&
_output_shapes
:x
'conv2dmpo_1/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
!conv2dmpo_1/Tensordot_1/Reshape_1Reshape'conv2dmpo_1/Tensordot_1/transpose_1:y:00conv2dmpo_1/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:?
conv2dmpo_1/Tensordot_1/MatMulMatMul(conv2dmpo_1/Tensordot_1/Reshape:output:0*conv2dmpo_1/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes

:@~
conv2dmpo_1/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  ?
conv2dmpo_1/Tensordot_1Reshape(conv2dmpo_1/Tensordot_1/MatMul:product:0&conv2dmpo_1/Tensordot_1/shape:output:0*
T0*.
_output_shapes
:?
&conv2dmpo_1/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
!conv2dmpo_1/Tensordot_2/transpose	Transposeconv2dmpo_1/Tensordot:output:0/conv2dmpo_1/Tensordot_2/transpose/perm:output:0*
T0*.
_output_shapes
:v
%conv2dmpo_1/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?      ?
conv2dmpo_1/Tensordot_2/ReshapeReshape%conv2dmpo_1/Tensordot_2/transpose:y:0.conv2dmpo_1/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes
:	??
(conv2dmpo_1/Tensordot_2/transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
#conv2dmpo_1/Tensordot_2/transpose_1	Transpose conv2dmpo_1/Tensordot_1:output:01conv2dmpo_1/Tensordot_2/transpose_1/perm:output:0*
T0*.
_output_shapes
:x
'conv2dmpo_1/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   ?
!conv2dmpo_1/Tensordot_2/Reshape_1Reshape'conv2dmpo_1/Tensordot_2/transpose_1:y:00conv2dmpo_1/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	??
conv2dmpo_1/Tensordot_2/MatMulMatMul(conv2dmpo_1/Tensordot_2/Reshape:output:0*conv2dmpo_1/Tensordot_2/Reshape_1:output:0*
T0* 
_output_shapes
:
???
conv2dmpo_1/Tensordot_2/shapeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                              ?
conv2dmpo_1/Tensordot_2Reshape(conv2dmpo_1/Tensordot_2/MatMul:product:0&conv2dmpo_1/Tensordot_2/shape:output:0*
T0*>
_output_shapes,
*:(?
conv2dmpo_1/transpose/permConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                   	            ?
conv2dmpo_1/transpose	Transpose conv2dmpo_1/Tensordot_2:output:0#conv2dmpo_1/transpose/perm:output:0*
T0*>
_output_shapes,
*:(?
conv2dmpo_1/transpose_1/permConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                	               ?
conv2dmpo_1/transpose_1	Transposeconv2dmpo_1/transpose:y:0%conv2dmpo_1/transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(?
conv2dmpo_1/ShapeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                              i
conv2dmpo_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:k
!conv2dmpo_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: k
!conv2dmpo_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2dmpo_1/strided_sliceStridedSliceconv2dmpo_1/Shape:output:0(conv2dmpo_1/strided_slice/stack:output:0*conv2dmpo_1/strided_slice/stack_1:output:0*conv2dmpo_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask[
conv2dmpo_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: y
conv2dmpo_1/ProdProd"conv2dmpo_1/strided_slice:output:0conv2dmpo_1/Const:output:0*
T0*
_output_shapes
: k
!conv2dmpo_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#conv2dmpo_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#conv2dmpo_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2dmpo_1/strided_slice_1StridedSliceconv2dmpo_1/Shape:output:0*conv2dmpo_1/strided_slice_1/stack:output:0,conv2dmpo_1/strided_slice_1/stack_1:output:0,conv2dmpo_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskl
conv2dmpo_1/concat/values_1Packconv2dmpo_1/Prod:output:0*
N*
T0*
_output_shapes
:b
conv2dmpo_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv2dmpo_1/concatConcatV2$conv2dmpo_1/strided_slice_1:output:0$conv2dmpo_1/concat/values_1:output:0 conv2dmpo_1/concat/axis:output:0*
N*
T0*
_output_shapes
:?
conv2dmpo_1/ReshapeReshapeconv2dmpo_1/transpose_1:y:0conv2dmpo_1/concat:output:0*
T0*2
_output_shapes 
:?
conv2dmpo_1/transpose_2/permConst*
_output_shapes
:*
dtype0*1
value(B&"                      ?
conv2dmpo_1/transpose_2	Transposeconv2dmpo_1/Reshape:output:0%conv2dmpo_1/transpose_2/perm:output:0*
T0*2
_output_shapes 
:x
conv2dmpo_1/Shape_1Const*
_output_shapes
:*
dtype0*1
value(B&"                     k
!conv2dmpo_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:m
#conv2dmpo_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: m
#conv2dmpo_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2dmpo_1/strided_slice_2StridedSliceconv2dmpo_1/Shape_1:output:0*conv2dmpo_1/strided_slice_2/stack:output:0,conv2dmpo_1/strided_slice_2/stack_1:output:0,conv2dmpo_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask]
conv2dmpo_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
conv2dmpo_1/Prod_1Prod$conv2dmpo_1/strided_slice_2:output:0conv2dmpo_1/Const_1:output:0*
T0*
_output_shapes
: k
!conv2dmpo_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#conv2dmpo_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#conv2dmpo_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2dmpo_1/strided_slice_3StridedSliceconv2dmpo_1/Shape_1:output:0*conv2dmpo_1/strided_slice_3/stack:output:0,conv2dmpo_1/strided_slice_3/stack_1:output:0,conv2dmpo_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
conv2dmpo_1/concat_1/values_1Packconv2dmpo_1/Prod_1:output:0*
N*
T0*
_output_shapes
:d
conv2dmpo_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv2dmpo_1/concat_1ConcatV2$conv2dmpo_1/strided_slice_3:output:0&conv2dmpo_1/concat_1/values_1:output:0"conv2dmpo_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
conv2dmpo_1/Reshape_1Reshapeconv2dmpo_1/transpose_2:y:0conv2dmpo_1/concat_1:output:0*
T0*'
_output_shapes
:??
conv2dmpo_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0conv2dmpo_1/Reshape_1:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
$conv2dmpo_1/Reshape_2/ReadVariableOpReadVariableOp-conv2dmpo_1_reshape_2_readvariableop_resource*
_output_shapes	
:?*
dtype0l
conv2dmpo_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
conv2dmpo_1/Reshape_2Reshape,conv2dmpo_1/Reshape_2/ReadVariableOp:value:0$conv2dmpo_1/Reshape_2/shape:output:0*
T0*
_output_shapes
:	??
conv2dmpo_1/addAddV2conv2dmpo_1/Conv2D:output:0conv2dmpo_1/Reshape_2:output:0*
T0*0
_output_shapes
:??????????h
conv2dmpo_1/ReluReluconv2dmpo_1/add:z:0*
T0*0
_output_shapes
:???????????
&conv2dmpo_1/ActivityRegularizer/SquareSquareconv2dmpo_1/Relu:activations:0*
T0*0
_output_shapes
:??????????~
%conv2dmpo_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
#conv2dmpo_1/ActivityRegularizer/SumSum*conv2dmpo_1/ActivityRegularizer/Square:y:0.conv2dmpo_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: j
%conv2dmpo_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#conv2dmpo_1/ActivityRegularizer/mulMul.conv2dmpo_1/ActivityRegularizer/mul/x:output:0,conv2dmpo_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: s
%conv2dmpo_1/ActivityRegularizer/ShapeShapeconv2dmpo_1/Relu:activations:0*
T0*
_output_shapes
:}
3conv2dmpo_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5conv2dmpo_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5conv2dmpo_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-conv2dmpo_1/ActivityRegularizer/strided_sliceStridedSlice.conv2dmpo_1/ActivityRegularizer/Shape:output:0<conv2dmpo_1/ActivityRegularizer/strided_slice/stack:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_1:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
$conv2dmpo_1/ActivityRegularizer/CastCast6conv2dmpo_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
'conv2dmpo_1/ActivityRegularizer/truedivRealDiv'conv2dmpo_1/ActivityRegularizer/mul:z:0(conv2dmpo_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
max_pooling2d_1/MaxPoolMaxPoolconv2dmpo_1/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
_
fruit_tn1/ShapeShape max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:g
fruit_tn1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
fruit_tn1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
fruit_tn1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
fruit_tn1/strided_sliceStridedSlicefruit_tn1/Shape:output:0&fruit_tn1/strided_slice/stack:output:0(fruit_tn1/strided_slice/stack_1:output:0(fruit_tn1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
fruit_tn1/Rank/packedPack fruit_tn1/strided_slice:output:0*
N*
T0*
_output_shapes
:P
fruit_tn1/RankConst*
_output_shapes
: *
dtype0*
value	B :W
fruit_tn1/range/startConst*
_output_shapes
: *
dtype0*
value	B : W
fruit_tn1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
fruit_tn1/rangeRangefruit_tn1/range/start:output:0fruit_tn1/Rank:output:0fruit_tn1/range/delta:output:0*
_output_shapes
:k
fruit_tn1/Max/inputPack fruit_tn1/strided_slice:output:0*
N*
T0*
_output_shapes
:m
fruit_tn1/MaxMaxfruit_tn1/Max/input:output:0fruit_tn1/range:output:0*
T0*
_output_shapes
: r
0fruit_tn1/loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : ?
*fruit_tn1/loop_body/PlaceholderWithDefaultPlaceholderWithDefault9fruit_tn1/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: i
fruit_tn1/loop_body/ShapeShape max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:q
'fruit_tn1/loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)fruit_tn1/loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)fruit_tn1/loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!fruit_tn1/loop_body/strided_sliceStridedSlice"fruit_tn1/loop_body/Shape:output:00fruit_tn1/loop_body/strided_slice/stack:output:02fruit_tn1/loop_body/strided_slice/stack_1:output:02fruit_tn1/loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
fruit_tn1/loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :?
fruit_tn1/loop_body/GreaterGreater*fruit_tn1/loop_body/strided_slice:output:0&fruit_tn1/loop_body/Greater/y:output:0*
T0*
_output_shapes
: `
fruit_tn1/loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : ?
fruit_tn1/loop_body/SelectV2SelectV2fruit_tn1/loop_body/Greater:z:03fruit_tn1/loop_body/PlaceholderWithDefault:output:0'fruit_tn1/loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: c
!fruit_tn1/loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
fruit_tn1/loop_body/GatherV2GatherV2 max_pooling2d_1/MaxPool:output:0%fruit_tn1/loop_body/SelectV2:output:0*fruit_tn1/loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:??
"fruit_tn1/loop_body/ReadVariableOpReadVariableOp+fruit_tn1_loop_body_readvariableop_resource*"
_output_shapes
:*
dtype0?
$fruit_tn1/loop_body/ReadVariableOp_1ReadVariableOp-fruit_tn1_loop_body_readvariableop_1_resource*'
_output_shapes
:?*
dtype0?
$fruit_tn1/loop_body/ReadVariableOp_2ReadVariableOp-fruit_tn1_loop_body_readvariableop_2_resource*"
_output_shapes
:*
dtype0?
,fruit_tn1/loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
'fruit_tn1/loop_body/Tensordot/transpose	Transpose%fruit_tn1/loop_body/GatherV2:output:05fruit_tn1/loop_body/Tensordot/transpose/perm:output:0*
T0*#
_output_shapes
:?|
+fruit_tn1/loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
%fruit_tn1/loop_body/Tensordot/ReshapeReshape+fruit_tn1/loop_body/Tensordot/transpose:y:04fruit_tn1/loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	??
.fruit_tn1/loop_body/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
)fruit_tn1/loop_body/Tensordot/transpose_1	Transpose*fruit_tn1/loop_body/ReadVariableOp:value:07fruit_tn1/loop_body/Tensordot/transpose_1/perm:output:0*
T0*"
_output_shapes
:~
-fruit_tn1/loop_body/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
'fruit_tn1/loop_body/Tensordot/Reshape_1Reshape-fruit_tn1/loop_body/Tensordot/transpose_1:y:06fruit_tn1/loop_body/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:?
$fruit_tn1/loop_body/Tensordot/MatMulMatMul.fruit_tn1/loop_body/Tensordot/Reshape:output:00fruit_tn1/loop_body/Tensordot/Reshape_1:output:0*
T0*
_output_shapes
:	?|
#fruit_tn1/loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            ?
fruit_tn1/loop_body/TensordotReshape.fruit_tn1/loop_body/Tensordot/MatMul:product:0,fruit_tn1/loop_body/Tensordot/shape:output:0*
T0*'
_output_shapes
:??
.fruit_tn1/loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
)fruit_tn1/loop_body/Tensordot_1/transpose	Transpose,fruit_tn1/loop_body/ReadVariableOp_2:value:07fruit_tn1/loop_body/Tensordot_1/transpose/perm:output:0*
T0*"
_output_shapes
:~
-fruit_tn1/loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
'fruit_tn1/loop_body/Tensordot_1/ReshapeReshape-fruit_tn1/loop_body/Tensordot_1/transpose:y:06fruit_tn1/loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:?
/fruit_tn1/loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
)fruit_tn1/loop_body/Tensordot_1/Reshape_1Reshape&fruit_tn1/loop_body/Tensordot:output:08fruit_tn1/loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?(?
&fruit_tn1/loop_body/Tensordot_1/MatMulMatMul0fruit_tn1/loop_body/Tensordot_1/Reshape:output:02fruit_tn1/loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	?(?
%fruit_tn1/loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"               ?
fruit_tn1/loop_body/Tensordot_1Reshape0fruit_tn1/loop_body/Tensordot_1/MatMul:product:0.fruit_tn1/loop_body/Tensordot_1/shape:output:0*
T0*+
_output_shapes
:?~
-fruit_tn1/loop_body/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
'fruit_tn1/loop_body/Tensordot_2/ReshapeReshape,fruit_tn1/loop_body/ReadVariableOp_1:value:06fruit_tn1/loop_body/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes
:	?2?
.fruit_tn1/loop_body/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
)fruit_tn1/loop_body/Tensordot_2/transpose	Transpose(fruit_tn1/loop_body/Tensordot_1:output:07fruit_tn1/loop_body/Tensordot_2/transpose/perm:output:0*
T0*+
_output_shapes
:??
/fruit_tn1/loop_body/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
)fruit_tn1/loop_body/Tensordot_2/Reshape_1Reshape-fruit_tn1/loop_body/Tensordot_2/transpose:y:08fruit_tn1/loop_body/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2?
&fruit_tn1/loop_body/Tensordot_2/MatMulMatMul0fruit_tn1/loop_body/Tensordot_2/Reshape:output:02fruit_tn1/loop_body/Tensordot_2/Reshape_1:output:0*
T0*
_output_shapes

:z
%fruit_tn1/loop_body/Tensordot_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
fruit_tn1/loop_body/Tensordot_2Reshape0fruit_tn1/loop_body/Tensordot_2/MatMul:product:0.fruit_tn1/loop_body/Tensordot_2/shape:output:0*
T0*"
_output_shapes
:w
"fruit_tn1/loop_body/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
fruit_tn1/loop_body/transpose	Transpose(fruit_tn1/loop_body/Tensordot_2:output:0+fruit_tn1/loop_body/transpose/perm:output:0*
T0*"
_output_shapes
:?
&fruit_tn1/loop_body/add/ReadVariableOpReadVariableOp/fruit_tn1_loop_body_add_readvariableop_resource*"
_output_shapes
:*
dtype0?
fruit_tn1/loop_body/addAddV2!fruit_tn1/loop_body/transpose:y:0.fruit_tn1/loop_body/add/ReadVariableOp:value:0*
T0*"
_output_shapes
:f
fruit_tn1/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
fruit_tn1/pfor/ReshapeReshapefruit_tn1/Max:output:0%fruit_tn1/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:\
fruit_tn1/pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : \
fruit_tn1/pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
fruit_tn1/pfor/rangeRange#fruit_tn1/pfor/range/start:output:0fruit_tn1/Max:output:0#fruit_tn1/pfor/range/delta:output:0*#
_output_shapes
:?????????h
&fruit_tn1/loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : i
'fruit_tn1/loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
%fruit_tn1/loop_body/SelectV2/pfor/addAddV2/fruit_tn1/loop_body/SelectV2/pfor/Rank:output:00fruit_tn1/loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: j
(fruit_tn1/loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :j
(fruit_tn1/loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : k
)fruit_tn1/loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
'fruit_tn1/loop_body/SelectV2/pfor/add_1AddV21fruit_tn1/loop_body/SelectV2/pfor/Rank_2:output:02fruit_tn1/loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: ?
)fruit_tn1/loop_body/SelectV2/pfor/MaximumMaximum1fruit_tn1/loop_body/SelectV2/pfor/Rank_1:output:0)fruit_tn1/loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: ?
+fruit_tn1/loop_body/SelectV2/pfor/Maximum_1Maximum+fruit_tn1/loop_body/SelectV2/pfor/add_1:z:0-fruit_tn1/loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: t
'fruit_tn1/loop_body/SelectV2/pfor/ShapeShapefruit_tn1/pfor/range:output:0*
T0*
_output_shapes
:?
%fruit_tn1/loop_body/SelectV2/pfor/subSub/fruit_tn1/loop_body/SelectV2/pfor/Maximum_1:z:01fruit_tn1/loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: y
/fruit_tn1/loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
)fruit_tn1/loop_body/SelectV2/pfor/ReshapeReshape)fruit_tn1/loop_body/SelectV2/pfor/sub:z:08fruit_tn1/loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:v
,fruit_tn1/loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:?
&fruit_tn1/loop_body/SelectV2/pfor/TileTile5fruit_tn1/loop_body/SelectV2/pfor/Tile/input:output:02fruit_tn1/loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: 
5fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/fruit_tn1/loop_body/SelectV2/pfor/strided_sliceStridedSlice0fruit_tn1/loop_body/SelectV2/pfor/Shape:output:0>fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack:output:0@fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_1:output:0@fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
7fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
9fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1StridedSlice0fruit_tn1/loop_body/SelectV2/pfor/Shape:output:0@fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stack:output:0Bfruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:0Bfruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_masko
-fruit_tn1/loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(fruit_tn1/loop_body/SelectV2/pfor/concatConcatV28fruit_tn1/loop_body/SelectV2/pfor/strided_slice:output:0/fruit_tn1/loop_body/SelectV2/pfor/Tile:output:0:fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1:output:06fruit_tn1/loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
+fruit_tn1/loop_body/SelectV2/pfor/Reshape_1Reshapefruit_tn1/pfor/range:output:01fruit_tn1/loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:??????????
*fruit_tn1/loop_body/SelectV2/pfor/SelectV2SelectV2fruit_tn1/loop_body/Greater:z:04fruit_tn1/loop_body/SelectV2/pfor/Reshape_1:output:0'fruit_tn1/loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:?????????q
/fruit_tn1/loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*fruit_tn1/loop_body/GatherV2/pfor/GatherV2GatherV2 max_pooling2d_1/MaxPool:output:03fruit_tn1/loop_body/SelectV2/pfor/SelectV2:output:08fruit_tn1/loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*0
_output_shapes
:??????????t
2fruit_tn1/loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
0fruit_tn1/loop_body/Tensordot/transpose/pfor/addAddV25fruit_tn1/loop_body/Tensordot/transpose/perm:output:0;fruit_tn1/loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:?
<fruit_tn1/loop_body/Tensordot/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: z
8fruit_tn1/loop_body/Tensordot/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
3fruit_tn1/loop_body/Tensordot/transpose/pfor/concatConcatV2Efruit_tn1/loop_body/Tensordot/transpose/pfor/concat/values_0:output:04fruit_tn1/loop_body/Tensordot/transpose/pfor/add:z:0Afruit_tn1/loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
6fruit_tn1/loop_body/Tensordot/transpose/pfor/Transpose	Transpose3fruit_tn1/loop_body/GatherV2/pfor/GatherV2:output:0<fruit_tn1/loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*0
_output_shapes
:??????????x
6fruit_tn1/loop_body/Tensordot/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1fruit_tn1/loop_body/Tensordot/Reshape/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:04fruit_tn1/loop_body/Tensordot/Reshape/shape:output:0?fruit_tn1/loop_body/Tensordot/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
2fruit_tn1/loop_body/Tensordot/Reshape/pfor/ReshapeReshape:fruit_tn1/loop_body/Tensordot/transpose/pfor/Transpose:y:0:fruit_tn1/loop_body/Tensordot/Reshape/pfor/concat:output:0*
T0*,
_output_shapes
:???????????
/fruit_tn1/loop_body/Tensordot/MatMul/pfor/ShapeShape;fruit_tn1/loop_body/Tensordot/Reshape/pfor/Reshape:output:0*
T0*
_output_shapes
:{
9fruit_tn1/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
/fruit_tn1/loop_body/Tensordot/MatMul/pfor/splitSplitBfruit_tn1/loop_body/Tensordot/MatMul/pfor/split/split_dim:output:08fruit_tn1/loop_body/Tensordot/MatMul/pfor/Shape:output:0*
T0*&
_output_shapes
:::*
	num_splitz
7fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB |
9fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
1fruit_tn1/loop_body/Tensordot/MatMul/pfor/ReshapeReshape8fruit_tn1/loop_body/Tensordot/MatMul/pfor/split:output:0Bfruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1:output:0*
T0*
_output_shapes
: |
9fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ~
;fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
3fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1Reshape8fruit_tn1/loop_body/Tensordot/MatMul/pfor/split:output:1Dfruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: |
9fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ~
;fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
3fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2Reshape8fruit_tn1/loop_body/Tensordot/MatMul/pfor/split:output:2Dfruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ?
-fruit_tn1/loop_body/Tensordot/MatMul/pfor/mulMul:fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape:output:0<fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: ?
9fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack1fruit_tn1/loop_body/Tensordot/MatMul/pfor/mul:z:0<fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:?
3fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape;fruit_tn1/loop_body/Tensordot/Reshape/pfor/Reshape:output:0Bfruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:???????????????????
0fruit_tn1/loop_body/Tensordot/MatMul/pfor/MatMulMatMul<fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3:output:00fruit_tn1/loop_body/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:??????????
;fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
??????????
9fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePack:fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape:output:0<fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0Dfruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:?
3fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4Reshape:fruit_tn1/loop_body/Tensordot/MatMul/pfor/MatMul:product:0Bfruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*,
_output_shapes
:??????????p
.fruit_tn1/loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)fruit_tn1/loop_body/Tensordot/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:0,fruit_tn1/loop_body/Tensordot/shape:output:07fruit_tn1/loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
*fruit_tn1/loop_body/Tensordot/pfor/ReshapeReshape<fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4:output:02fruit_tn1/loop_body/Tensordot/pfor/concat:output:0*
T0*4
_output_shapes"
 :??????????|
:fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
5fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:08fruit_tn1/loop_body/Tensordot_1/Reshape_1/shape:output:0Cfruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
6fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape3fruit_tn1/loop_body/Tensordot/pfor/Reshape:output:0>fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:??????????(?
1fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/ShapeShape?fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:?
?fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSlice:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Hfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSlice:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/MinimumMinimumBfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0Dfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: u
3fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :?
1fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/EqualEqual7fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Minimum:z:0<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: ?
4fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          ?
4fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          ?
2fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/SelectSelect5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Equal:z:0=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/t:output:0=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:?
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSlice:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSlice:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSlice:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/stackPackDfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0Dfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0Dfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose	Transpose?fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'????????????????????????????
3fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/ReshapeReshape9fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose:y:0:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:??????????(?
3fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape_1Shape<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:}
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/splitSplitDfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split/split_dim:output:0<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split~
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1Reshape:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split:output:0Ffruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2Reshape:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split:output:1Ffruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3Reshape:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split:output:2Ffruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ?
/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/mulMul>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ?
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePack>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:03fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:?
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4Reshape<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0Dfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:???????????????????
2fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul0fruit_tn1/loop_body/Tensordot_1/Reshape:output:0>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:??????????
=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
??????????
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePackFfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:0>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:?
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5Reshape<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0Dfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'????????????????????????????
<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
7fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1	Transpose>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0Efruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????(r
0fruit_tn1/loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+fruit_tn1/loop_body/Tensordot_1/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:0.fruit_tn1/loop_body/Tensordot_1/shape:output:09fruit_tn1/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,fruit_tn1/loop_body/Tensordot_1/pfor/ReshapeReshape;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:04fruit_tn1/loop_body/Tensordot_1/pfor/concat:output:0*
T0*8
_output_shapes&
$:"??????????v
4fruit_tn1/loop_body/Tensordot_2/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
2fruit_tn1/loop_body/Tensordot_2/transpose/pfor/addAddV27fruit_tn1/loop_body/Tensordot_2/transpose/perm:output:0=fruit_tn1/loop_body/Tensordot_2/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:?
>fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: |
:fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
5fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concatConcatV2Gfruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/values_0:output:06fruit_tn1/loop_body/Tensordot_2/transpose/pfor/add:z:0Cfruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
8fruit_tn1/loop_body/Tensordot_2/transpose/pfor/Transpose	Transpose5fruit_tn1/loop_body/Tensordot_1/pfor/Reshape:output:0>fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat:output:0*
T0*8
_output_shapes&
$:"??????????|
:fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
5fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:08fruit_tn1/loop_body/Tensordot_2/Reshape_1/shape:output:0Cfruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
6fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/ReshapeReshape<fruit_tn1/loop_body/Tensordot_2/transpose/pfor/Transpose:y:0>fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:??????????2?
1fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/ShapeShape?fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:?
?fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_sliceStridedSlice:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Hfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1StridedSlice:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/MinimumMinimumBfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice:output:0Dfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: u
3fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :?
1fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/EqualEqual7fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Minimum:z:0<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: ?
4fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          ?
4fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          ?
2fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/SelectSelect5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Equal:z:0=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/t:output:0=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:?
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2StridedSlice:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3StridedSlice:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4StridedSlice:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/stackPackDfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2:output:0Dfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3:output:0Dfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose	Transpose?fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'????????????????????????????
3fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/ReshapeReshape9fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose:y:0:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:?2??????????
3fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape_1Shape<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:}
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/splitSplitDfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split/split_dim:output:0<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split~
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1Reshape:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split:output:0Ffruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2Reshape:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split:output:1Ffruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3Reshape:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split:output:2Ffruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ?
/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/mulMul>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ?
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shapePack>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1:output:03fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:?
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4Reshape<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0Dfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:???????????????????
2fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/MatMulMatMul0fruit_tn1/loop_body/Tensordot_2/Reshape:output:0>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:??????????
=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
??????????
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shapePackFfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0:output:0>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:?
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5Reshape<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/MatMul:product:0Dfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'????????????????????????????
<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
7fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1	Transpose>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5:output:0Efruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????r
0fruit_tn1/loop_body/Tensordot_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+fruit_tn1/loop_body/Tensordot_2/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:0.fruit_tn1/loop_body/Tensordot_2/shape:output:09fruit_tn1/loop_body/Tensordot_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,fruit_tn1/loop_body/Tensordot_2/pfor/ReshapeReshape;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1:y:04fruit_tn1/loop_body/Tensordot_2/pfor/concat:output:0*
T0*/
_output_shapes
:?????????j
(fruit_tn1/loop_body/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
&fruit_tn1/loop_body/transpose/pfor/addAddV2+fruit_tn1/loop_body/transpose/perm:output:01fruit_tn1/loop_body/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:|
2fruit_tn1/loop_body/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: p
.fruit_tn1/loop_body/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)fruit_tn1/loop_body/transpose/pfor/concatConcatV2;fruit_tn1/loop_body/transpose/pfor/concat/values_0:output:0*fruit_tn1/loop_body/transpose/pfor/add:z:07fruit_tn1/loop_body/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,fruit_tn1/loop_body/transpose/pfor/Transpose	Transpose5fruit_tn1/loop_body/Tensordot_2/pfor/Reshape:output:02fruit_tn1/loop_body/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:?????????c
!fruit_tn1/loop_body/add/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :e
#fruit_tn1/loop_body/add/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :d
"fruit_tn1/loop_body/add/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
 fruit_tn1/loop_body/add/pfor/addAddV2,fruit_tn1/loop_body/add/pfor/Rank_1:output:0+fruit_tn1/loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: ?
$fruit_tn1/loop_body/add/pfor/MaximumMaximum$fruit_tn1/loop_body/add/pfor/add:z:0*fruit_tn1/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: ?
"fruit_tn1/loop_body/add/pfor/ShapeShape0fruit_tn1/loop_body/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:?
 fruit_tn1/loop_body/add/pfor/subSub(fruit_tn1/loop_body/add/pfor/Maximum:z:0*fruit_tn1/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: t
*fruit_tn1/loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
$fruit_tn1/loop_body/add/pfor/ReshapeReshape$fruit_tn1/loop_body/add/pfor/sub:z:03fruit_tn1/loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:q
'fruit_tn1/loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:?
!fruit_tn1/loop_body/add/pfor/TileTile0fruit_tn1/loop_body/add/pfor/Tile/input:output:0-fruit_tn1/loop_body/add/pfor/Reshape:output:0*
T0*
_output_shapes
: z
0fruit_tn1/loop_body/add/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2fruit_tn1/loop_body/add/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2fruit_tn1/loop_body/add/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*fruit_tn1/loop_body/add/pfor/strided_sliceStridedSlice+fruit_tn1/loop_body/add/pfor/Shape:output:09fruit_tn1/loop_body/add/pfor/strided_slice/stack:output:0;fruit_tn1/loop_body/add/pfor/strided_slice/stack_1:output:0;fruit_tn1/loop_body/add/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
2fruit_tn1/loop_body/add/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:~
4fruit_tn1/loop_body/add/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ~
4fruit_tn1/loop_body/add/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,fruit_tn1/loop_body/add/pfor/strided_slice_1StridedSlice+fruit_tn1/loop_body/add/pfor/Shape:output:0;fruit_tn1/loop_body/add/pfor/strided_slice_1/stack:output:0=fruit_tn1/loop_body/add/pfor/strided_slice_1/stack_1:output:0=fruit_tn1/loop_body/add/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskj
(fruit_tn1/loop_body/add/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
#fruit_tn1/loop_body/add/pfor/concatConcatV23fruit_tn1/loop_body/add/pfor/strided_slice:output:0*fruit_tn1/loop_body/add/pfor/Tile:output:05fruit_tn1/loop_body/add/pfor/strided_slice_1:output:01fruit_tn1/loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
&fruit_tn1/loop_body/add/pfor/Reshape_1Reshape0fruit_tn1/loop_body/transpose/pfor/Transpose:y:0,fruit_tn1/loop_body/add/pfor/concat:output:0*
T0*/
_output_shapes
:??????????
"fruit_tn1/loop_body/add/pfor/AddV2AddV2/fruit_tn1/loop_body/add/pfor/Reshape_1:output:0.fruit_tn1/loop_body/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????h
fruit_tn1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
fruit_tn1/ReshapeReshape&fruit_tn1/loop_body/add/pfor/AddV2:z:0 fruit_tn1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????@d
fruit_tn1/ReluRelufruit_tn1/Reshape:output:0*
T0*'
_output_shapes
:?????????@~
$fruit_tn1/ActivityRegularizer/SquareSquarefruit_tn1/Relu:activations:0*
T0*'
_output_shapes
:?????????@t
#fruit_tn1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
!fruit_tn1/ActivityRegularizer/SumSum(fruit_tn1/ActivityRegularizer/Square:y:0,fruit_tn1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: h
#fruit_tn1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
!fruit_tn1/ActivityRegularizer/mulMul,fruit_tn1/ActivityRegularizer/mul/x:output:0*fruit_tn1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: o
#fruit_tn1/ActivityRegularizer/ShapeShapefruit_tn1/Relu:activations:0*
T0*
_output_shapes
:{
1fruit_tn1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3fruit_tn1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3fruit_tn1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+fruit_tn1/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn1/ActivityRegularizer/Shape:output:0:fruit_tn1/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"fruit_tn1/ActivityRegularizer/CastCast4fruit_tn1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%fruit_tn1/ActivityRegularizer/truedivRealDiv%fruit_tn1/ActivityRegularizer/mul:z:0&fruit_tn1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: m
dropout1/IdentityIdentityfruit_tn1/Relu:activations:0*
T0*'
_output_shapes
:?????????@Y
fruit_tn2/ShapeShapedropout1/Identity:output:0*
T0*
_output_shapes
:g
fruit_tn2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
fruit_tn2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
fruit_tn2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
fruit_tn2/strided_sliceStridedSlicefruit_tn2/Shape:output:0&fruit_tn2/strided_slice/stack:output:0(fruit_tn2/strided_slice/stack_1:output:0(fruit_tn2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
fruit_tn2/Rank/packedPack fruit_tn2/strided_slice:output:0*
N*
T0*
_output_shapes
:P
fruit_tn2/RankConst*
_output_shapes
: *
dtype0*
value	B :W
fruit_tn2/range/startConst*
_output_shapes
: *
dtype0*
value	B : W
fruit_tn2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
fruit_tn2/rangeRangefruit_tn2/range/start:output:0fruit_tn2/Rank:output:0fruit_tn2/range/delta:output:0*
_output_shapes
:k
fruit_tn2/Max/inputPack fruit_tn2/strided_slice:output:0*
N*
T0*
_output_shapes
:m
fruit_tn2/MaxMaxfruit_tn2/Max/input:output:0fruit_tn2/range:output:0*
T0*
_output_shapes
: r
0fruit_tn2/loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : ?
*fruit_tn2/loop_body/PlaceholderWithDefaultPlaceholderWithDefault9fruit_tn2/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: c
fruit_tn2/loop_body/ShapeShapedropout1/Identity:output:0*
T0*
_output_shapes
:q
'fruit_tn2/loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)fruit_tn2/loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)fruit_tn2/loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!fruit_tn2/loop_body/strided_sliceStridedSlice"fruit_tn2/loop_body/Shape:output:00fruit_tn2/loop_body/strided_slice/stack:output:02fruit_tn2/loop_body/strided_slice/stack_1:output:02fruit_tn2/loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
fruit_tn2/loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :?
fruit_tn2/loop_body/GreaterGreater*fruit_tn2/loop_body/strided_slice:output:0&fruit_tn2/loop_body/Greater/y:output:0*
T0*
_output_shapes
: `
fruit_tn2/loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : ?
fruit_tn2/loop_body/SelectV2SelectV2fruit_tn2/loop_body/Greater:z:03fruit_tn2/loop_body/PlaceholderWithDefault:output:0'fruit_tn2/loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: c
!fruit_tn2/loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
fruit_tn2/loop_body/GatherV2GatherV2dropout1/Identity:output:0%fruit_tn2/loop_body/SelectV2:output:0*fruit_tn2/loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:@v
!fruit_tn2/loop_body/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
fruit_tn2/loop_body/ReshapeReshape%fruit_tn2/loop_body/GatherV2:output:0*fruit_tn2/loop_body/Reshape/shape:output:0*
T0*"
_output_shapes
:?
"fruit_tn2/loop_body/ReadVariableOpReadVariableOp+fruit_tn2_loop_body_readvariableop_resource*
_output_shapes

:*
dtype0?
$fruit_tn2/loop_body/ReadVariableOp_1ReadVariableOp-fruit_tn2_loop_body_readvariableop_1_resource*'
_output_shapes
:?*
dtype0?
$fruit_tn2/loop_body/ReadVariableOp_2ReadVariableOp-fruit_tn2_loop_body_readvariableop_2_resource*
_output_shapes

:*
dtype0?
,fruit_tn2/loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
'fruit_tn2/loop_body/Tensordot/transpose	Transpose$fruit_tn2/loop_body/Reshape:output:05fruit_tn2/loop_body/Tensordot/transpose/perm:output:0*
T0*"
_output_shapes
:|
+fruit_tn2/loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
%fruit_tn2/loop_body/Tensordot/ReshapeReshape+fruit_tn2/loop_body/Tensordot/transpose:y:04fruit_tn2/loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:?
$fruit_tn2/loop_body/Tensordot/MatMulMatMul.fruit_tn2/loop_body/Tensordot/Reshape:output:0*fruit_tn2/loop_body/ReadVariableOp:value:0*
T0*
_output_shapes

:x
#fruit_tn2/loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
fruit_tn2/loop_body/TensordotReshape.fruit_tn2/loop_body/Tensordot/MatMul:product:0,fruit_tn2/loop_body/Tensordot/shape:output:0*
T0*"
_output_shapes
:?
.fruit_tn2/loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
)fruit_tn2/loop_body/Tensordot_1/transpose	Transpose,fruit_tn2/loop_body/ReadVariableOp_1:value:07fruit_tn2/loop_body/Tensordot_1/transpose/perm:output:0*
T0*'
_output_shapes
:?~
-fruit_tn2/loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"     ?
'fruit_tn2/loop_body/Tensordot_1/ReshapeReshape-fruit_tn2/loop_body/Tensordot_1/transpose:y:06fruit_tn2/loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes
:	??
0fruit_tn2/loop_body/Tensordot_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
+fruit_tn2/loop_body/Tensordot_1/transpose_1	Transpose&fruit_tn2/loop_body/Tensordot:output:09fruit_tn2/loop_body/Tensordot_1/transpose_1/perm:output:0*
T0*"
_output_shapes
:?
/fruit_tn2/loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
)fruit_tn2/loop_body/Tensordot_1/Reshape_1Reshape/fruit_tn2/loop_body/Tensordot_1/transpose_1:y:08fruit_tn2/loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:?
&fruit_tn2/loop_body/Tensordot_1/MatMulMatMul0fruit_tn2/loop_body/Tensordot_1/Reshape:output:02fruit_tn2/loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	?z
%fruit_tn2/loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"?         ?
fruit_tn2/loop_body/Tensordot_1Reshape0fruit_tn2/loop_body/Tensordot_1/MatMul:product:0.fruit_tn2/loop_body/Tensordot_1/shape:output:0*
T0*#
_output_shapes
:?~
-fruit_tn2/loop_body/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
'fruit_tn2/loop_body/Tensordot_2/ReshapeReshape,fruit_tn2/loop_body/ReadVariableOp_2:value:06fruit_tn2/loop_body/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes

:?
.fruit_tn2/loop_body/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
)fruit_tn2/loop_body/Tensordot_2/transpose	Transpose(fruit_tn2/loop_body/Tensordot_1:output:07fruit_tn2/loop_body/Tensordot_2/transpose/perm:output:0*
T0*#
_output_shapes
:??
/fruit_tn2/loop_body/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   ?
)fruit_tn2/loop_body/Tensordot_2/Reshape_1Reshape-fruit_tn2/loop_body/Tensordot_2/transpose:y:08fruit_tn2/loop_body/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	??
&fruit_tn2/loop_body/Tensordot_2/MatMulMatMul0fruit_tn2/loop_body/Tensordot_2/Reshape:output:02fruit_tn2/loop_body/Tensordot_2/Reshape_1:output:0*
T0*
_output_shapes
:	?p
%fruit_tn2/loop_body/Tensordot_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:??
fruit_tn2/loop_body/Tensordot_2Reshape0fruit_tn2/loop_body/Tensordot_2/MatMul:product:0.fruit_tn2/loop_body/Tensordot_2/shape:output:0*
T0*
_output_shapes	
:??
&fruit_tn2/loop_body/add/ReadVariableOpReadVariableOp/fruit_tn2_loop_body_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
fruit_tn2/loop_body/addAddV2(fruit_tn2/loop_body/Tensordot_2:output:0.fruit_tn2/loop_body/add/ReadVariableOp:value:0*
T0*
_output_shapes	
:?f
fruit_tn2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
fruit_tn2/pfor/ReshapeReshapefruit_tn2/Max:output:0%fruit_tn2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:\
fruit_tn2/pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : \
fruit_tn2/pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
fruit_tn2/pfor/rangeRange#fruit_tn2/pfor/range/start:output:0fruit_tn2/Max:output:0#fruit_tn2/pfor/range/delta:output:0*#
_output_shapes
:?????????h
&fruit_tn2/loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : i
'fruit_tn2/loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
%fruit_tn2/loop_body/SelectV2/pfor/addAddV2/fruit_tn2/loop_body/SelectV2/pfor/Rank:output:00fruit_tn2/loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: j
(fruit_tn2/loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :j
(fruit_tn2/loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : k
)fruit_tn2/loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
'fruit_tn2/loop_body/SelectV2/pfor/add_1AddV21fruit_tn2/loop_body/SelectV2/pfor/Rank_2:output:02fruit_tn2/loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: ?
)fruit_tn2/loop_body/SelectV2/pfor/MaximumMaximum1fruit_tn2/loop_body/SelectV2/pfor/Rank_1:output:0)fruit_tn2/loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: ?
+fruit_tn2/loop_body/SelectV2/pfor/Maximum_1Maximum+fruit_tn2/loop_body/SelectV2/pfor/add_1:z:0-fruit_tn2/loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: t
'fruit_tn2/loop_body/SelectV2/pfor/ShapeShapefruit_tn2/pfor/range:output:0*
T0*
_output_shapes
:?
%fruit_tn2/loop_body/SelectV2/pfor/subSub/fruit_tn2/loop_body/SelectV2/pfor/Maximum_1:z:01fruit_tn2/loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: y
/fruit_tn2/loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
)fruit_tn2/loop_body/SelectV2/pfor/ReshapeReshape)fruit_tn2/loop_body/SelectV2/pfor/sub:z:08fruit_tn2/loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:v
,fruit_tn2/loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:?
&fruit_tn2/loop_body/SelectV2/pfor/TileTile5fruit_tn2/loop_body/SelectV2/pfor/Tile/input:output:02fruit_tn2/loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: 
5fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/fruit_tn2/loop_body/SelectV2/pfor/strided_sliceStridedSlice0fruit_tn2/loop_body/SelectV2/pfor/Shape:output:0>fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack:output:0@fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_1:output:0@fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
7fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
9fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1StridedSlice0fruit_tn2/loop_body/SelectV2/pfor/Shape:output:0@fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stack:output:0Bfruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:0Bfruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_masko
-fruit_tn2/loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(fruit_tn2/loop_body/SelectV2/pfor/concatConcatV28fruit_tn2/loop_body/SelectV2/pfor/strided_slice:output:0/fruit_tn2/loop_body/SelectV2/pfor/Tile:output:0:fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1:output:06fruit_tn2/loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
+fruit_tn2/loop_body/SelectV2/pfor/Reshape_1Reshapefruit_tn2/pfor/range:output:01fruit_tn2/loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:??????????
*fruit_tn2/loop_body/SelectV2/pfor/SelectV2SelectV2fruit_tn2/loop_body/Greater:z:04fruit_tn2/loop_body/SelectV2/pfor/Reshape_1:output:0'fruit_tn2/loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:?????????q
/fruit_tn2/loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*fruit_tn2/loop_body/GatherV2/pfor/GatherV2GatherV2dropout1/Identity:output:03fruit_tn2/loop_body/SelectV2/pfor/SelectV2:output:08fruit_tn2/loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:?????????@n
,fruit_tn2/loop_body/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'fruit_tn2/loop_body/Reshape/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:0*fruit_tn2/loop_body/Reshape/shape:output:05fruit_tn2/loop_body/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
(fruit_tn2/loop_body/Reshape/pfor/ReshapeReshape3fruit_tn2/loop_body/GatherV2/pfor/GatherV2:output:00fruit_tn2/loop_body/Reshape/pfor/concat:output:0*
T0*/
_output_shapes
:?????????t
2fruit_tn2/loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
0fruit_tn2/loop_body/Tensordot/transpose/pfor/addAddV25fruit_tn2/loop_body/Tensordot/transpose/perm:output:0;fruit_tn2/loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:?
<fruit_tn2/loop_body/Tensordot/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: z
8fruit_tn2/loop_body/Tensordot/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
3fruit_tn2/loop_body/Tensordot/transpose/pfor/concatConcatV2Efruit_tn2/loop_body/Tensordot/transpose/pfor/concat/values_0:output:04fruit_tn2/loop_body/Tensordot/transpose/pfor/add:z:0Afruit_tn2/loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
6fruit_tn2/loop_body/Tensordot/transpose/pfor/Transpose	Transpose1fruit_tn2/loop_body/Reshape/pfor/Reshape:output:0<fruit_tn2/loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:?????????x
6fruit_tn2/loop_body/Tensordot/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1fruit_tn2/loop_body/Tensordot/Reshape/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:04fruit_tn2/loop_body/Tensordot/Reshape/shape:output:0?fruit_tn2/loop_body/Tensordot/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
2fruit_tn2/loop_body/Tensordot/Reshape/pfor/ReshapeReshape:fruit_tn2/loop_body/Tensordot/transpose/pfor/Transpose:y:0:fruit_tn2/loop_body/Tensordot/Reshape/pfor/concat:output:0*
T0*+
_output_shapes
:??????????
/fruit_tn2/loop_body/Tensordot/MatMul/pfor/ShapeShape;fruit_tn2/loop_body/Tensordot/Reshape/pfor/Reshape:output:0*
T0*
_output_shapes
:{
9fruit_tn2/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
/fruit_tn2/loop_body/Tensordot/MatMul/pfor/splitSplitBfruit_tn2/loop_body/Tensordot/MatMul/pfor/split/split_dim:output:08fruit_tn2/loop_body/Tensordot/MatMul/pfor/Shape:output:0*
T0*&
_output_shapes
:::*
	num_splitz
7fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB |
9fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
1fruit_tn2/loop_body/Tensordot/MatMul/pfor/ReshapeReshape8fruit_tn2/loop_body/Tensordot/MatMul/pfor/split:output:0Bfruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1:output:0*
T0*
_output_shapes
: |
9fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ~
;fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
3fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1Reshape8fruit_tn2/loop_body/Tensordot/MatMul/pfor/split:output:1Dfruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: |
9fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ~
;fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
3fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2Reshape8fruit_tn2/loop_body/Tensordot/MatMul/pfor/split:output:2Dfruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ?
-fruit_tn2/loop_body/Tensordot/MatMul/pfor/mulMul:fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape:output:0<fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: ?
9fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack1fruit_tn2/loop_body/Tensordot/MatMul/pfor/mul:z:0<fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:?
3fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape;fruit_tn2/loop_body/Tensordot/Reshape/pfor/Reshape:output:0Bfruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:???????????????????
0fruit_tn2/loop_body/Tensordot/MatMul/pfor/MatMulMatMul<fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0*fruit_tn2/loop_body/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
;fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
??????????
9fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePack:fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape:output:0<fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0Dfruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:?
3fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4Reshape:fruit_tn2/loop_body/Tensordot/MatMul/pfor/MatMul:product:0Bfruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*+
_output_shapes
:?????????p
.fruit_tn2/loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)fruit_tn2/loop_body/Tensordot/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:0,fruit_tn2/loop_body/Tensordot/shape:output:07fruit_tn2/loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
*fruit_tn2/loop_body/Tensordot/pfor/ReshapeReshape<fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4:output:02fruit_tn2/loop_body/Tensordot/pfor/concat:output:0*
T0*/
_output_shapes
:?????????x
6fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
4fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/addAddV29fruit_tn2/loop_body/Tensordot_1/transpose_1/perm:output:0?fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/add/y:output:0*
T0*
_output_shapes
:?
@fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: ~
<fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
7fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concatConcatV2Ifruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/values_0:output:08fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/add:z:0Efruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
:fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/Transpose	Transpose3fruit_tn2/loop_body/Tensordot/pfor/Reshape:output:0@fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat:output:0*
T0*/
_output_shapes
:?????????|
:fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
5fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:08fruit_tn2/loop_body/Tensordot_1/Reshape_1/shape:output:0Cfruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
6fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape>fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/Transpose:y:0>fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*+
_output_shapes
:??????????
1fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/ShapeShape?fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:?
?fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSlice:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Hfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSlice:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/MinimumMinimumBfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0Dfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: u
3fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :?
1fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/EqualEqual7fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Minimum:z:0<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: ?
4fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          ?
4fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          ?
2fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/SelectSelect5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Equal:z:0=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/t:output:0=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:?
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSlice:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSlice:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSlice:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/stackPackDfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0Dfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0Dfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose	Transpose?fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'????????????????????????????
3fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/ReshapeReshape9fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose:y:0:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*+
_output_shapes
:??????????
3fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape_1Shape<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:}
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/splitSplitDfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split/split_dim:output:0<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split~
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1Reshape:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split:output:0Ffruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2Reshape:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split:output:1Ffruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3Reshape:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split:output:2Ffruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ?
/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/mulMul>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ?
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePack>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:03fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:?
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4Reshape<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0Dfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:???????????????????
2fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul0fruit_tn2/loop_body/Tensordot_1/Reshape:output:0>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*(
_output_shapes
:???????????
=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
??????????
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePackFfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:0>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:?
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5Reshape<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0Dfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'????????????????????????????
<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
7fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1	Transpose>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0Efruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????r
0fruit_tn2/loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+fruit_tn2/loop_body/Tensordot_1/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:0.fruit_tn2/loop_body/Tensordot_1/shape:output:09fruit_tn2/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,fruit_tn2/loop_body/Tensordot_1/pfor/ReshapeReshape;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:04fruit_tn2/loop_body/Tensordot_1/pfor/concat:output:0*
T0*0
_output_shapes
:??????????v
4fruit_tn2/loop_body/Tensordot_2/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
2fruit_tn2/loop_body/Tensordot_2/transpose/pfor/addAddV27fruit_tn2/loop_body/Tensordot_2/transpose/perm:output:0=fruit_tn2/loop_body/Tensordot_2/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:?
>fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: |
:fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
5fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concatConcatV2Gfruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/values_0:output:06fruit_tn2/loop_body/Tensordot_2/transpose/pfor/add:z:0Cfruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
8fruit_tn2/loop_body/Tensordot_2/transpose/pfor/Transpose	Transpose5fruit_tn2/loop_body/Tensordot_1/pfor/Reshape:output:0>fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat:output:0*
T0*0
_output_shapes
:??????????|
:fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
5fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:08fruit_tn2/loop_body/Tensordot_2/Reshape_1/shape:output:0Cfruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
6fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/ReshapeReshape<fruit_tn2/loop_body/Tensordot_2/transpose/pfor/Transpose:y:0>fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:???????????
1fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/ShapeShape?fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:?
?fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_sliceStridedSlice:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Hfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1StridedSlice:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/MinimumMinimumBfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice:output:0Dfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: u
3fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :?
1fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/EqualEqual7fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Minimum:z:0<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: ?
4fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          ?
4fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          ?
2fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/SelectSelect5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Equal:z:0=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/t:output:0=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:?
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2StridedSlice:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3StridedSlice:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4StridedSlice:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/stackPackDfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2:output:0Dfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3:output:0Dfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose	Transpose?fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'????????????????????????????
3fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/ReshapeReshape9fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose:y:0:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:???????????
3fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape_1Shape<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:}
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
1fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/splitSplitDfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split/split_dim:output:0<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split~
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1Reshape:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split:output:0Ffruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2Reshape:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split:output:1Ffruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3Reshape:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split:output:2Ffruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ?
/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/mulMul>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ?
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shapePack>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1:output:03fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:?
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4Reshape<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0Dfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:???????????????????
2fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/MatMulMatMul0fruit_tn2/loop_body/Tensordot_2/Reshape:output:0>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:??????????
=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
??????????
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shapePackFfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0:output:0>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:?
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5Reshape<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/MatMul:product:0Dfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'????????????????????????????
<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
7fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1	Transpose>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5:output:0Efruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????r
0fruit_tn2/loop_body/Tensordot_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+fruit_tn2/loop_body/Tensordot_2/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:0.fruit_tn2/loop_body/Tensordot_2/shape:output:09fruit_tn2/loop_body/Tensordot_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,fruit_tn2/loop_body/Tensordot_2/pfor/ReshapeReshape;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1:y:04fruit_tn2/loop_body/Tensordot_2/pfor/concat:output:0*
T0*(
_output_shapes
:??????????c
!fruit_tn2/loop_body/add/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :e
#fruit_tn2/loop_body/add/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :d
"fruit_tn2/loop_body/add/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
 fruit_tn2/loop_body/add/pfor/addAddV2,fruit_tn2/loop_body/add/pfor/Rank_1:output:0+fruit_tn2/loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: ?
$fruit_tn2/loop_body/add/pfor/MaximumMaximum$fruit_tn2/loop_body/add/pfor/add:z:0*fruit_tn2/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: ?
"fruit_tn2/loop_body/add/pfor/ShapeShape5fruit_tn2/loop_body/Tensordot_2/pfor/Reshape:output:0*
T0*
_output_shapes
:?
 fruit_tn2/loop_body/add/pfor/subSub(fruit_tn2/loop_body/add/pfor/Maximum:z:0*fruit_tn2/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: t
*fruit_tn2/loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
$fruit_tn2/loop_body/add/pfor/ReshapeReshape$fruit_tn2/loop_body/add/pfor/sub:z:03fruit_tn2/loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:q
'fruit_tn2/loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:?
!fruit_tn2/loop_body/add/pfor/TileTile0fruit_tn2/loop_body/add/pfor/Tile/input:output:0-fruit_tn2/loop_body/add/pfor/Reshape:output:0*
T0*
_output_shapes
: z
0fruit_tn2/loop_body/add/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2fruit_tn2/loop_body/add/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2fruit_tn2/loop_body/add/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*fruit_tn2/loop_body/add/pfor/strided_sliceStridedSlice+fruit_tn2/loop_body/add/pfor/Shape:output:09fruit_tn2/loop_body/add/pfor/strided_slice/stack:output:0;fruit_tn2/loop_body/add/pfor/strided_slice/stack_1:output:0;fruit_tn2/loop_body/add/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
2fruit_tn2/loop_body/add/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:~
4fruit_tn2/loop_body/add/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ~
4fruit_tn2/loop_body/add/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,fruit_tn2/loop_body/add/pfor/strided_slice_1StridedSlice+fruit_tn2/loop_body/add/pfor/Shape:output:0;fruit_tn2/loop_body/add/pfor/strided_slice_1/stack:output:0=fruit_tn2/loop_body/add/pfor/strided_slice_1/stack_1:output:0=fruit_tn2/loop_body/add/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskj
(fruit_tn2/loop_body/add/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
#fruit_tn2/loop_body/add/pfor/concatConcatV23fruit_tn2/loop_body/add/pfor/strided_slice:output:0*fruit_tn2/loop_body/add/pfor/Tile:output:05fruit_tn2/loop_body/add/pfor/strided_slice_1:output:01fruit_tn2/loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
&fruit_tn2/loop_body/add/pfor/Reshape_1Reshape5fruit_tn2/loop_body/Tensordot_2/pfor/Reshape:output:0,fruit_tn2/loop_body/add/pfor/concat:output:0*
T0*(
_output_shapes
:???????????
"fruit_tn2/loop_body/add/pfor/AddV2AddV2/fruit_tn2/loop_body/add/pfor/Reshape_1:output:0.fruit_tn2/loop_body/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????h
fruit_tn2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
fruit_tn2/ReshapeReshape&fruit_tn2/loop_body/add/pfor/AddV2:z:0 fruit_tn2/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????k
fruit_tn2/SoftmaxSoftmaxfruit_tn2/Reshape:output:0*
T0*(
_output_shapes
:??????????~
$fruit_tn2/ActivityRegularizer/SquareSquarefruit_tn2/Softmax:softmax:0*
T0*(
_output_shapes
:??????????t
#fruit_tn2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
!fruit_tn2/ActivityRegularizer/SumSum(fruit_tn2/ActivityRegularizer/Square:y:0,fruit_tn2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: h
#fruit_tn2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
!fruit_tn2/ActivityRegularizer/mulMul,fruit_tn2/ActivityRegularizer/mul/x:output:0*fruit_tn2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: n
#fruit_tn2/ActivityRegularizer/ShapeShapefruit_tn2/Softmax:softmax:0*
T0*
_output_shapes
:{
1fruit_tn2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3fruit_tn2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3fruit_tn2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+fruit_tn2/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn2/ActivityRegularizer/Shape:output:0:fruit_tn2/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"fruit_tn2/ActivityRegularizer/CastCast4fruit_tn2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%fruit_tn2/ActivityRegularizer/truedivRealDiv%fruit_tn2/ActivityRegularizer/mul:z:0&fruit_tn2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: k
IdentityIdentityfruit_tn2/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:??????????i

Identity_1Identity)conv2dmpo/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: k

Identity_2Identity+conv2dmpo_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_3Identity)fruit_tn1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_4Identity)fruit_tn2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^conv2dmpo/ReadVariableOp^conv2dmpo/ReadVariableOp_1#^conv2dmpo/Reshape_2/ReadVariableOp^conv2dmpo_1/ReadVariableOp^conv2dmpo_1/ReadVariableOp_1^conv2dmpo_1/ReadVariableOp_2^conv2dmpo_1/ReadVariableOp_3%^conv2dmpo_1/Reshape_2/ReadVariableOp#^fruit_tn1/loop_body/ReadVariableOp%^fruit_tn1/loop_body/ReadVariableOp_1%^fruit_tn1/loop_body/ReadVariableOp_2'^fruit_tn1/loop_body/add/ReadVariableOp#^fruit_tn2/loop_body/ReadVariableOp%^fruit_tn2/loop_body/ReadVariableOp_1%^fruit_tn2/loop_body/ReadVariableOp_2'^fruit_tn2/loop_body/add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????22: : : : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp24
conv2dmpo/ReadVariableOpconv2dmpo/ReadVariableOp28
conv2dmpo/ReadVariableOp_1conv2dmpo/ReadVariableOp_12H
"conv2dmpo/Reshape_2/ReadVariableOp"conv2dmpo/Reshape_2/ReadVariableOp28
conv2dmpo_1/ReadVariableOpconv2dmpo_1/ReadVariableOp2<
conv2dmpo_1/ReadVariableOp_1conv2dmpo_1/ReadVariableOp_12<
conv2dmpo_1/ReadVariableOp_2conv2dmpo_1/ReadVariableOp_22<
conv2dmpo_1/ReadVariableOp_3conv2dmpo_1/ReadVariableOp_32L
$conv2dmpo_1/Reshape_2/ReadVariableOp$conv2dmpo_1/Reshape_2/ReadVariableOp2H
"fruit_tn1/loop_body/ReadVariableOp"fruit_tn1/loop_body/ReadVariableOp2L
$fruit_tn1/loop_body/ReadVariableOp_1$fruit_tn1/loop_body/ReadVariableOp_12L
$fruit_tn1/loop_body/ReadVariableOp_2$fruit_tn1/loop_body/ReadVariableOp_22P
&fruit_tn1/loop_body/add/ReadVariableOp&fruit_tn1/loop_body/add/ReadVariableOp2H
"fruit_tn2/loop_body/ReadVariableOp"fruit_tn2/loop_body/ReadVariableOp2L
$fruit_tn2/loop_body/ReadVariableOp_1$fruit_tn2/loop_body/ReadVariableOp_12L
$fruit_tn2/loop_body/ReadVariableOp_2$fruit_tn2/loop_body/ReadVariableOp_22P
&fruit_tn2/loop_body/add/ReadVariableOp&fruit_tn2/loop_body/add/ReadVariableOp:W S
/
_output_shapes
:?????????22
 
_user_specified_nameinputs
??	
?
!__inference__wrapped_model_412277
input_12L
2sequential_1_conv2d_conv2d_readvariableop_resource:A
3sequential_1_conv2d_biasadd_readvariableop_resource:H
.sequential_1_conv2dmpo_readvariableop_resource:J
0sequential_1_conv2dmpo_readvariableop_1_resource:F
8sequential_1_conv2dmpo_reshape_2_readvariableop_resource:J
0sequential_1_conv2dmpo_1_readvariableop_resource:L
2sequential_1_conv2dmpo_1_readvariableop_1_resource:L
2sequential_1_conv2dmpo_1_readvariableop_2_resource:L
2sequential_1_conv2dmpo_1_readvariableop_3_resource:I
:sequential_1_conv2dmpo_1_reshape_2_readvariableop_resource:	?N
8sequential_1_fruit_tn1_loop_body_readvariableop_resource:U
:sequential_1_fruit_tn1_loop_body_readvariableop_1_resource:?P
:sequential_1_fruit_tn1_loop_body_readvariableop_2_resource:R
<sequential_1_fruit_tn1_loop_body_add_readvariableop_resource:J
8sequential_1_fruit_tn2_loop_body_readvariableop_resource:U
:sequential_1_fruit_tn2_loop_body_readvariableop_1_resource:?L
:sequential_1_fruit_tn2_loop_body_readvariableop_2_resource:K
<sequential_1_fruit_tn2_loop_body_add_readvariableop_resource:	?
identity??*sequential_1/conv2d/BiasAdd/ReadVariableOp?)sequential_1/conv2d/Conv2D/ReadVariableOp?%sequential_1/conv2dmpo/ReadVariableOp?'sequential_1/conv2dmpo/ReadVariableOp_1?/sequential_1/conv2dmpo/Reshape_2/ReadVariableOp?'sequential_1/conv2dmpo_1/ReadVariableOp?)sequential_1/conv2dmpo_1/ReadVariableOp_1?)sequential_1/conv2dmpo_1/ReadVariableOp_2?)sequential_1/conv2dmpo_1/ReadVariableOp_3?1sequential_1/conv2dmpo_1/Reshape_2/ReadVariableOp?/sequential_1/fruit_tn1/loop_body/ReadVariableOp?1sequential_1/fruit_tn1/loop_body/ReadVariableOp_1?1sequential_1/fruit_tn1/loop_body/ReadVariableOp_2?3sequential_1/fruit_tn1/loop_body/add/ReadVariableOp?/sequential_1/fruit_tn2/loop_body/ReadVariableOp?1sequential_1/fruit_tn2/loop_body/ReadVariableOp_1?1sequential_1/fruit_tn2/loop_body/ReadVariableOp_2?3sequential_1/fruit_tn2/loop_body/add/ReadVariableOp?
)sequential_1/conv2d/Conv2D/ReadVariableOpReadVariableOp2sequential_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_1/conv2d/Conv2DConv2Dinput_121sequential_1/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????//*
paddingVALID*
strides
?
*sequential_1/conv2d/BiasAdd/ReadVariableOpReadVariableOp3sequential_1_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_1/conv2d/BiasAddBiasAdd#sequential_1/conv2d/Conv2D:output:02sequential_1/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????//?
%sequential_1/conv2dmpo/ReadVariableOpReadVariableOp.sequential_1_conv2dmpo_readvariableop_resource*&
_output_shapes
:*
dtype0?
'sequential_1/conv2dmpo/ReadVariableOp_1ReadVariableOp0sequential_1_conv2dmpo_readvariableop_1_resource*&
_output_shapes
:*
dtype0?
/sequential_1/conv2dmpo/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
*sequential_1/conv2dmpo/Tensordot/transpose	Transpose/sequential_1/conv2dmpo/ReadVariableOp_1:value:08sequential_1/conv2dmpo/Tensordot/transpose/perm:output:0*
T0*&
_output_shapes
:
.sequential_1/conv2dmpo/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
(sequential_1/conv2dmpo/Tensordot/ReshapeReshape.sequential_1/conv2dmpo/Tensordot/transpose:y:07sequential_1/conv2dmpo/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:?
1sequential_1/conv2dmpo/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
,sequential_1/conv2dmpo/Tensordot/transpose_1	Transpose-sequential_1/conv2dmpo/ReadVariableOp:value:0:sequential_1/conv2dmpo/Tensordot/transpose_1/perm:output:0*
T0*&
_output_shapes
:?
0sequential_1/conv2dmpo/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
*sequential_1/conv2dmpo/Tensordot/Reshape_1Reshape0sequential_1/conv2dmpo/Tensordot/transpose_1:y:09sequential_1/conv2dmpo/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:?
'sequential_1/conv2dmpo/Tensordot/MatMulMatMul1sequential_1/conv2dmpo/Tensordot/Reshape:output:03sequential_1/conv2dmpo/Tensordot/Reshape_1:output:0*
T0*
_output_shapes

:?
&sequential_1/conv2dmpo/Tensordot/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  ?
 sequential_1/conv2dmpo/TensordotReshape1sequential_1/conv2dmpo/Tensordot/MatMul:product:0/sequential_1/conv2dmpo/Tensordot/shape:output:0*
T0*.
_output_shapes
:?
%sequential_1/conv2dmpo/transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
 sequential_1/conv2dmpo/transpose	Transpose)sequential_1/conv2dmpo/Tensordot:output:0.sequential_1/conv2dmpo/transpose/perm:output:0*
T0*.
_output_shapes
:?
'sequential_1/conv2dmpo/transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
"sequential_1/conv2dmpo/transpose_1	Transpose$sequential_1/conv2dmpo/transpose:y:00sequential_1/conv2dmpo/transpose_1/perm:output:0*
T0*.
_output_shapes
:}
sequential_1/conv2dmpo/ShapeConst*
_output_shapes
:*
dtype0*-
value$B""                  t
*sequential_1/conv2dmpo/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,sequential_1/conv2dmpo/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,sequential_1/conv2dmpo/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$sequential_1/conv2dmpo/strided_sliceStridedSlice%sequential_1/conv2dmpo/Shape:output:03sequential_1/conv2dmpo/strided_slice/stack:output:05sequential_1/conv2dmpo/strided_slice/stack_1:output:05sequential_1/conv2dmpo/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskf
sequential_1/conv2dmpo/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
sequential_1/conv2dmpo/ProdProd-sequential_1/conv2dmpo/strided_slice:output:0%sequential_1/conv2dmpo/Const:output:0*
T0*
_output_shapes
: v
,sequential_1/conv2dmpo/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_1/conv2dmpo/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_1/conv2dmpo/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&sequential_1/conv2dmpo/strided_slice_1StridedSlice%sequential_1/conv2dmpo/Shape:output:05sequential_1/conv2dmpo/strided_slice_1/stack:output:07sequential_1/conv2dmpo/strided_slice_1/stack_1:output:07sequential_1/conv2dmpo/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
&sequential_1/conv2dmpo/concat/values_1Pack$sequential_1/conv2dmpo/Prod:output:0*
N*
T0*
_output_shapes
:m
"sequential_1/conv2dmpo/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
sequential_1/conv2dmpo/concatConcatV2/sequential_1/conv2dmpo/strided_slice_1:output:0/sequential_1/conv2dmpo/concat/values_1:output:0+sequential_1/conv2dmpo/concat/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_1/conv2dmpo/ReshapeReshape&sequential_1/conv2dmpo/transpose_1:y:0&sequential_1/conv2dmpo/concat:output:0*
T0**
_output_shapes
:?
'sequential_1/conv2dmpo/transpose_2/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
"sequential_1/conv2dmpo/transpose_2	Transpose'sequential_1/conv2dmpo/Reshape:output:00sequential_1/conv2dmpo/transpose_2/perm:output:0*
T0**
_output_shapes
:{
sequential_1/conv2dmpo/Shape_1Const*
_output_shapes
:*
dtype0*)
value B"               v
,sequential_1/conv2dmpo/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.sequential_1/conv2dmpo/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.sequential_1/conv2dmpo/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&sequential_1/conv2dmpo/strided_slice_2StridedSlice'sequential_1/conv2dmpo/Shape_1:output:05sequential_1/conv2dmpo/strided_slice_2/stack:output:07sequential_1/conv2dmpo/strided_slice_2/stack_1:output:07sequential_1/conv2dmpo/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskh
sequential_1/conv2dmpo/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
sequential_1/conv2dmpo/Prod_1Prod/sequential_1/conv2dmpo/strided_slice_2:output:0'sequential_1/conv2dmpo/Const_1:output:0*
T0*
_output_shapes
: v
,sequential_1/conv2dmpo/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_1/conv2dmpo/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_1/conv2dmpo/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&sequential_1/conv2dmpo/strided_slice_3StridedSlice'sequential_1/conv2dmpo/Shape_1:output:05sequential_1/conv2dmpo/strided_slice_3/stack:output:07sequential_1/conv2dmpo/strided_slice_3/stack_1:output:07sequential_1/conv2dmpo/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
(sequential_1/conv2dmpo/concat_1/values_1Pack&sequential_1/conv2dmpo/Prod_1:output:0*
N*
T0*
_output_shapes
:o
$sequential_1/conv2dmpo/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
sequential_1/conv2dmpo/concat_1ConcatV2/sequential_1/conv2dmpo/strided_slice_3:output:01sequential_1/conv2dmpo/concat_1/values_1:output:0-sequential_1/conv2dmpo/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
 sequential_1/conv2dmpo/Reshape_1Reshape&sequential_1/conv2dmpo/transpose_2:y:0(sequential_1/conv2dmpo/concat_1:output:0*
T0*&
_output_shapes
:?
sequential_1/conv2dmpo/Conv2DConv2D$sequential_1/conv2d/BiasAdd:output:0)sequential_1/conv2dmpo/Reshape_1:output:0*
T0*/
_output_shapes
:?????????//*
paddingSAME*
strides
?
/sequential_1/conv2dmpo/Reshape_2/ReadVariableOpReadVariableOp8sequential_1_conv2dmpo_reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0w
&sequential_1/conv2dmpo/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
 sequential_1/conv2dmpo/Reshape_2Reshape7sequential_1/conv2dmpo/Reshape_2/ReadVariableOp:value:0/sequential_1/conv2dmpo/Reshape_2/shape:output:0*
T0*
_output_shapes

:?
sequential_1/conv2dmpo/addAddV2&sequential_1/conv2dmpo/Conv2D:output:0)sequential_1/conv2dmpo/Reshape_2:output:0*
T0*/
_output_shapes
:?????????//}
sequential_1/conv2dmpo/ReluRelusequential_1/conv2dmpo/add:z:0*
T0*/
_output_shapes
:?????????//?
1sequential_1/conv2dmpo/ActivityRegularizer/SquareSquare)sequential_1/conv2dmpo/Relu:activations:0*
T0*/
_output_shapes
:?????????//?
0sequential_1/conv2dmpo/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
.sequential_1/conv2dmpo/ActivityRegularizer/SumSum5sequential_1/conv2dmpo/ActivityRegularizer/Square:y:09sequential_1/conv2dmpo/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: u
0sequential_1/conv2dmpo/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
.sequential_1/conv2dmpo/ActivityRegularizer/mulMul9sequential_1/conv2dmpo/ActivityRegularizer/mul/x:output:07sequential_1/conv2dmpo/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: ?
0sequential_1/conv2dmpo/ActivityRegularizer/ShapeShape)sequential_1/conv2dmpo/Relu:activations:0*
T0*
_output_shapes
:?
>sequential_1/conv2dmpo/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
@sequential_1/conv2dmpo/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@sequential_1/conv2dmpo/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_1/conv2dmpo/ActivityRegularizer/strided_sliceStridedSlice9sequential_1/conv2dmpo/ActivityRegularizer/Shape:output:0Gsequential_1/conv2dmpo/ActivityRegularizer/strided_slice/stack:output:0Isequential_1/conv2dmpo/ActivityRegularizer/strided_slice/stack_1:output:0Isequential_1/conv2dmpo/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
/sequential_1/conv2dmpo/ActivityRegularizer/CastCastAsequential_1/conv2dmpo/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
2sequential_1/conv2dmpo/ActivityRegularizer/truedivRealDiv2sequential_1/conv2dmpo/ActivityRegularizer/mul:z:03sequential_1/conv2dmpo/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
"sequential_1/max_pooling2d/MaxPoolMaxPool)sequential_1/conv2dmpo/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
?
'sequential_1/conv2dmpo_1/ReadVariableOpReadVariableOp0sequential_1_conv2dmpo_1_readvariableop_resource*&
_output_shapes
:*
dtype0?
)sequential_1/conv2dmpo_1/ReadVariableOp_1ReadVariableOp2sequential_1_conv2dmpo_1_readvariableop_1_resource*&
_output_shapes
:*
dtype0?
)sequential_1/conv2dmpo_1/ReadVariableOp_2ReadVariableOp2sequential_1_conv2dmpo_1_readvariableop_2_resource*&
_output_shapes
:*
dtype0?
)sequential_1/conv2dmpo_1/ReadVariableOp_3ReadVariableOp2sequential_1_conv2dmpo_1_readvariableop_3_resource*&
_output_shapes
:*
dtype0?
1sequential_1/conv2dmpo_1/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
,sequential_1/conv2dmpo_1/Tensordot/transpose	Transpose1sequential_1/conv2dmpo_1/ReadVariableOp_1:value:0:sequential_1/conv2dmpo_1/Tensordot/transpose/perm:output:0*
T0*&
_output_shapes
:?
0sequential_1/conv2dmpo_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@      ?
*sequential_1/conv2dmpo_1/Tensordot/ReshapeReshape0sequential_1/conv2dmpo_1/Tensordot/transpose:y:09sequential_1/conv2dmpo_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@?
3sequential_1/conv2dmpo_1/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
.sequential_1/conv2dmpo_1/Tensordot/transpose_1	Transpose/sequential_1/conv2dmpo_1/ReadVariableOp:value:0<sequential_1/conv2dmpo_1/Tensordot/transpose_1/perm:output:0*
T0*&
_output_shapes
:?
2sequential_1/conv2dmpo_1/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
,sequential_1/conv2dmpo_1/Tensordot/Reshape_1Reshape2sequential_1/conv2dmpo_1/Tensordot/transpose_1:y:0;sequential_1/conv2dmpo_1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:?
)sequential_1/conv2dmpo_1/Tensordot/MatMulMatMul3sequential_1/conv2dmpo_1/Tensordot/Reshape:output:05sequential_1/conv2dmpo_1/Tensordot/Reshape_1:output:0*
T0*
_output_shapes

:@?
(sequential_1/conv2dmpo_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  ?
"sequential_1/conv2dmpo_1/TensordotReshape3sequential_1/conv2dmpo_1/Tensordot/MatMul:product:01sequential_1/conv2dmpo_1/Tensordot/shape:output:0*
T0*.
_output_shapes
:?
3sequential_1/conv2dmpo_1/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
.sequential_1/conv2dmpo_1/Tensordot_1/transpose	Transpose1sequential_1/conv2dmpo_1/ReadVariableOp_2:value:0<sequential_1/conv2dmpo_1/Tensordot_1/transpose/perm:output:0*
T0*&
_output_shapes
:?
2sequential_1/conv2dmpo_1/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@      ?
,sequential_1/conv2dmpo_1/Tensordot_1/ReshapeReshape2sequential_1/conv2dmpo_1/Tensordot_1/transpose:y:0;sequential_1/conv2dmpo_1/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:@?
5sequential_1/conv2dmpo_1/Tensordot_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
0sequential_1/conv2dmpo_1/Tensordot_1/transpose_1	Transpose1sequential_1/conv2dmpo_1/ReadVariableOp_3:value:0>sequential_1/conv2dmpo_1/Tensordot_1/transpose_1/perm:output:0*
T0*&
_output_shapes
:?
4sequential_1/conv2dmpo_1/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
.sequential_1/conv2dmpo_1/Tensordot_1/Reshape_1Reshape4sequential_1/conv2dmpo_1/Tensordot_1/transpose_1:y:0=sequential_1/conv2dmpo_1/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:?
+sequential_1/conv2dmpo_1/Tensordot_1/MatMulMatMul5sequential_1/conv2dmpo_1/Tensordot_1/Reshape:output:07sequential_1/conv2dmpo_1/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes

:@?
*sequential_1/conv2dmpo_1/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  ?
$sequential_1/conv2dmpo_1/Tensordot_1Reshape5sequential_1/conv2dmpo_1/Tensordot_1/MatMul:product:03sequential_1/conv2dmpo_1/Tensordot_1/shape:output:0*
T0*.
_output_shapes
:?
3sequential_1/conv2dmpo_1/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
.sequential_1/conv2dmpo_1/Tensordot_2/transpose	Transpose+sequential_1/conv2dmpo_1/Tensordot:output:0<sequential_1/conv2dmpo_1/Tensordot_2/transpose/perm:output:0*
T0*.
_output_shapes
:?
2sequential_1/conv2dmpo_1/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?      ?
,sequential_1/conv2dmpo_1/Tensordot_2/ReshapeReshape2sequential_1/conv2dmpo_1/Tensordot_2/transpose:y:0;sequential_1/conv2dmpo_1/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes
:	??
5sequential_1/conv2dmpo_1/Tensordot_2/transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
0sequential_1/conv2dmpo_1/Tensordot_2/transpose_1	Transpose-sequential_1/conv2dmpo_1/Tensordot_1:output:0>sequential_1/conv2dmpo_1/Tensordot_2/transpose_1/perm:output:0*
T0*.
_output_shapes
:?
4sequential_1/conv2dmpo_1/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   ?
.sequential_1/conv2dmpo_1/Tensordot_2/Reshape_1Reshape4sequential_1/conv2dmpo_1/Tensordot_2/transpose_1:y:0=sequential_1/conv2dmpo_1/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	??
+sequential_1/conv2dmpo_1/Tensordot_2/MatMulMatMul5sequential_1/conv2dmpo_1/Tensordot_2/Reshape:output:07sequential_1/conv2dmpo_1/Tensordot_2/Reshape_1:output:0*
T0* 
_output_shapes
:
???
*sequential_1/conv2dmpo_1/Tensordot_2/shapeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                              ?
$sequential_1/conv2dmpo_1/Tensordot_2Reshape5sequential_1/conv2dmpo_1/Tensordot_2/MatMul:product:03sequential_1/conv2dmpo_1/Tensordot_2/shape:output:0*
T0*>
_output_shapes,
*:(?
'sequential_1/conv2dmpo_1/transpose/permConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   ?
"sequential_1/conv2dmpo_1/transpose	Transpose-sequential_1/conv2dmpo_1/Tensordot_2:output:00sequential_1/conv2dmpo_1/transpose/perm:output:0*
T0*>
_output_shapes,
*:(?
)sequential_1/conv2dmpo_1/transpose_1/permConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                	               ?
$sequential_1/conv2dmpo_1/transpose_1	Transpose&sequential_1/conv2dmpo_1/transpose:y:02sequential_1/conv2dmpo_1/transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(?
sequential_1/conv2dmpo_1/ShapeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                              v
,sequential_1/conv2dmpo_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.sequential_1/conv2dmpo_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.sequential_1/conv2dmpo_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&sequential_1/conv2dmpo_1/strided_sliceStridedSlice'sequential_1/conv2dmpo_1/Shape:output:05sequential_1/conv2dmpo_1/strided_slice/stack:output:07sequential_1/conv2dmpo_1/strided_slice/stack_1:output:07sequential_1/conv2dmpo_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskh
sequential_1/conv2dmpo_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
sequential_1/conv2dmpo_1/ProdProd/sequential_1/conv2dmpo_1/strided_slice:output:0'sequential_1/conv2dmpo_1/Const:output:0*
T0*
_output_shapes
: x
.sequential_1/conv2dmpo_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0sequential_1/conv2dmpo_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0sequential_1/conv2dmpo_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(sequential_1/conv2dmpo_1/strided_slice_1StridedSlice'sequential_1/conv2dmpo_1/Shape:output:07sequential_1/conv2dmpo_1/strided_slice_1/stack:output:09sequential_1/conv2dmpo_1/strided_slice_1/stack_1:output:09sequential_1/conv2dmpo_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
(sequential_1/conv2dmpo_1/concat/values_1Pack&sequential_1/conv2dmpo_1/Prod:output:0*
N*
T0*
_output_shapes
:o
$sequential_1/conv2dmpo_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
sequential_1/conv2dmpo_1/concatConcatV21sequential_1/conv2dmpo_1/strided_slice_1:output:01sequential_1/conv2dmpo_1/concat/values_1:output:0-sequential_1/conv2dmpo_1/concat/axis:output:0*
N*
T0*
_output_shapes
:?
 sequential_1/conv2dmpo_1/ReshapeReshape(sequential_1/conv2dmpo_1/transpose_1:y:0(sequential_1/conv2dmpo_1/concat:output:0*
T0*2
_output_shapes 
:?
)sequential_1/conv2dmpo_1/transpose_2/permConst*
_output_shapes
:*
dtype0*1
value(B&"                      ?
$sequential_1/conv2dmpo_1/transpose_2	Transpose)sequential_1/conv2dmpo_1/Reshape:output:02sequential_1/conv2dmpo_1/transpose_2/perm:output:0*
T0*2
_output_shapes 
:?
 sequential_1/conv2dmpo_1/Shape_1Const*
_output_shapes
:*
dtype0*1
value(B&"                     x
.sequential_1/conv2dmpo_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0sequential_1/conv2dmpo_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: z
0sequential_1/conv2dmpo_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(sequential_1/conv2dmpo_1/strided_slice_2StridedSlice)sequential_1/conv2dmpo_1/Shape_1:output:07sequential_1/conv2dmpo_1/strided_slice_2/stack:output:09sequential_1/conv2dmpo_1/strided_slice_2/stack_1:output:09sequential_1/conv2dmpo_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskj
 sequential_1/conv2dmpo_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
sequential_1/conv2dmpo_1/Prod_1Prod1sequential_1/conv2dmpo_1/strided_slice_2:output:0)sequential_1/conv2dmpo_1/Const_1:output:0*
T0*
_output_shapes
: x
.sequential_1/conv2dmpo_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0sequential_1/conv2dmpo_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0sequential_1/conv2dmpo_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(sequential_1/conv2dmpo_1/strided_slice_3StridedSlice)sequential_1/conv2dmpo_1/Shape_1:output:07sequential_1/conv2dmpo_1/strided_slice_3/stack:output:09sequential_1/conv2dmpo_1/strided_slice_3/stack_1:output:09sequential_1/conv2dmpo_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
*sequential_1/conv2dmpo_1/concat_1/values_1Pack(sequential_1/conv2dmpo_1/Prod_1:output:0*
N*
T0*
_output_shapes
:q
&sequential_1/conv2dmpo_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
!sequential_1/conv2dmpo_1/concat_1ConcatV21sequential_1/conv2dmpo_1/strided_slice_3:output:03sequential_1/conv2dmpo_1/concat_1/values_1:output:0/sequential_1/conv2dmpo_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
"sequential_1/conv2dmpo_1/Reshape_1Reshape(sequential_1/conv2dmpo_1/transpose_2:y:0*sequential_1/conv2dmpo_1/concat_1:output:0*
T0*'
_output_shapes
:??
sequential_1/conv2dmpo_1/Conv2DConv2D+sequential_1/max_pooling2d/MaxPool:output:0+sequential_1/conv2dmpo_1/Reshape_1:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
1sequential_1/conv2dmpo_1/Reshape_2/ReadVariableOpReadVariableOp:sequential_1_conv2dmpo_1_reshape_2_readvariableop_resource*
_output_shapes	
:?*
dtype0y
(sequential_1/conv2dmpo_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
"sequential_1/conv2dmpo_1/Reshape_2Reshape9sequential_1/conv2dmpo_1/Reshape_2/ReadVariableOp:value:01sequential_1/conv2dmpo_1/Reshape_2/shape:output:0*
T0*
_output_shapes
:	??
sequential_1/conv2dmpo_1/addAddV2(sequential_1/conv2dmpo_1/Conv2D:output:0+sequential_1/conv2dmpo_1/Reshape_2:output:0*
T0*0
_output_shapes
:???????????
sequential_1/conv2dmpo_1/ReluRelu sequential_1/conv2dmpo_1/add:z:0*
T0*0
_output_shapes
:???????????
3sequential_1/conv2dmpo_1/ActivityRegularizer/SquareSquare+sequential_1/conv2dmpo_1/Relu:activations:0*
T0*0
_output_shapes
:???????????
2sequential_1/conv2dmpo_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
0sequential_1/conv2dmpo_1/ActivityRegularizer/SumSum7sequential_1/conv2dmpo_1/ActivityRegularizer/Square:y:0;sequential_1/conv2dmpo_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: w
2sequential_1/conv2dmpo_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
0sequential_1/conv2dmpo_1/ActivityRegularizer/mulMul;sequential_1/conv2dmpo_1/ActivityRegularizer/mul/x:output:09sequential_1/conv2dmpo_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: ?
2sequential_1/conv2dmpo_1/ActivityRegularizer/ShapeShape+sequential_1/conv2dmpo_1/Relu:activations:0*
T0*
_output_shapes
:?
@sequential_1/conv2dmpo_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Bsequential_1/conv2dmpo_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bsequential_1/conv2dmpo_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:sequential_1/conv2dmpo_1/ActivityRegularizer/strided_sliceStridedSlice;sequential_1/conv2dmpo_1/ActivityRegularizer/Shape:output:0Isequential_1/conv2dmpo_1/ActivityRegularizer/strided_slice/stack:output:0Ksequential_1/conv2dmpo_1/ActivityRegularizer/strided_slice/stack_1:output:0Ksequential_1/conv2dmpo_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1sequential_1/conv2dmpo_1/ActivityRegularizer/CastCastCsequential_1/conv2dmpo_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
4sequential_1/conv2dmpo_1/ActivityRegularizer/truedivRealDiv4sequential_1/conv2dmpo_1/ActivityRegularizer/mul:z:05sequential_1/conv2dmpo_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
$sequential_1/max_pooling2d_1/MaxPoolMaxPool+sequential_1/conv2dmpo_1/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
y
sequential_1/fruit_tn1/ShapeShape-sequential_1/max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:t
*sequential_1/fruit_tn1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_1/fruit_tn1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_1/fruit_tn1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$sequential_1/fruit_tn1/strided_sliceStridedSlice%sequential_1/fruit_tn1/Shape:output:03sequential_1/fruit_tn1/strided_slice/stack:output:05sequential_1/fruit_tn1/strided_slice/stack_1:output:05sequential_1/fruit_tn1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"sequential_1/fruit_tn1/Rank/packedPack-sequential_1/fruit_tn1/strided_slice:output:0*
N*
T0*
_output_shapes
:]
sequential_1/fruit_tn1/RankConst*
_output_shapes
: *
dtype0*
value	B :d
"sequential_1/fruit_tn1/range/startConst*
_output_shapes
: *
dtype0*
value	B : d
"sequential_1/fruit_tn1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
sequential_1/fruit_tn1/rangeRange+sequential_1/fruit_tn1/range/start:output:0$sequential_1/fruit_tn1/Rank:output:0+sequential_1/fruit_tn1/range/delta:output:0*
_output_shapes
:?
 sequential_1/fruit_tn1/Max/inputPack-sequential_1/fruit_tn1/strided_slice:output:0*
N*
T0*
_output_shapes
:?
sequential_1/fruit_tn1/MaxMax)sequential_1/fruit_tn1/Max/input:output:0%sequential_1/fruit_tn1/range:output:0*
T0*
_output_shapes
: 
=sequential_1/fruit_tn1/loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : ?
7sequential_1/fruit_tn1/loop_body/PlaceholderWithDefaultPlaceholderWithDefaultFsequential_1/fruit_tn1/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: ?
&sequential_1/fruit_tn1/loop_body/ShapeShape-sequential_1/max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:~
4sequential_1/fruit_tn1/loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6sequential_1/fruit_tn1/loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6sequential_1/fruit_tn1/loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.sequential_1/fruit_tn1/loop_body/strided_sliceStridedSlice/sequential_1/fruit_tn1/loop_body/Shape:output:0=sequential_1/fruit_tn1/loop_body/strided_slice/stack:output:0?sequential_1/fruit_tn1/loop_body/strided_slice/stack_1:output:0?sequential_1/fruit_tn1/loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*sequential_1/fruit_tn1/loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :?
(sequential_1/fruit_tn1/loop_body/GreaterGreater7sequential_1/fruit_tn1/loop_body/strided_slice:output:03sequential_1/fruit_tn1/loop_body/Greater/y:output:0*
T0*
_output_shapes
: m
+sequential_1/fruit_tn1/loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_1/fruit_tn1/loop_body/SelectV2SelectV2,sequential_1/fruit_tn1/loop_body/Greater:z:0@sequential_1/fruit_tn1/loop_body/PlaceholderWithDefault:output:04sequential_1/fruit_tn1/loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: p
.sequential_1/fruit_tn1/loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_1/fruit_tn1/loop_body/GatherV2GatherV2-sequential_1/max_pooling2d_1/MaxPool:output:02sequential_1/fruit_tn1/loop_body/SelectV2:output:07sequential_1/fruit_tn1/loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:??
/sequential_1/fruit_tn1/loop_body/ReadVariableOpReadVariableOp8sequential_1_fruit_tn1_loop_body_readvariableop_resource*"
_output_shapes
:*
dtype0?
1sequential_1/fruit_tn1/loop_body/ReadVariableOp_1ReadVariableOp:sequential_1_fruit_tn1_loop_body_readvariableop_1_resource*'
_output_shapes
:?*
dtype0?
1sequential_1/fruit_tn1/loop_body/ReadVariableOp_2ReadVariableOp:sequential_1_fruit_tn1_loop_body_readvariableop_2_resource*"
_output_shapes
:*
dtype0?
9sequential_1/fruit_tn1/loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
4sequential_1/fruit_tn1/loop_body/Tensordot/transpose	Transpose2sequential_1/fruit_tn1/loop_body/GatherV2:output:0Bsequential_1/fruit_tn1/loop_body/Tensordot/transpose/perm:output:0*
T0*#
_output_shapes
:??
8sequential_1/fruit_tn1/loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
2sequential_1/fruit_tn1/loop_body/Tensordot/ReshapeReshape8sequential_1/fruit_tn1/loop_body/Tensordot/transpose:y:0Asequential_1/fruit_tn1/loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	??
;sequential_1/fruit_tn1/loop_body/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
6sequential_1/fruit_tn1/loop_body/Tensordot/transpose_1	Transpose7sequential_1/fruit_tn1/loop_body/ReadVariableOp:value:0Dsequential_1/fruit_tn1/loop_body/Tensordot/transpose_1/perm:output:0*
T0*"
_output_shapes
:?
:sequential_1/fruit_tn1/loop_body/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
4sequential_1/fruit_tn1/loop_body/Tensordot/Reshape_1Reshape:sequential_1/fruit_tn1/loop_body/Tensordot/transpose_1:y:0Csequential_1/fruit_tn1/loop_body/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:?
1sequential_1/fruit_tn1/loop_body/Tensordot/MatMulMatMul;sequential_1/fruit_tn1/loop_body/Tensordot/Reshape:output:0=sequential_1/fruit_tn1/loop_body/Tensordot/Reshape_1:output:0*
T0*
_output_shapes
:	??
0sequential_1/fruit_tn1/loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            ?
*sequential_1/fruit_tn1/loop_body/TensordotReshape;sequential_1/fruit_tn1/loop_body/Tensordot/MatMul:product:09sequential_1/fruit_tn1/loop_body/Tensordot/shape:output:0*
T0*'
_output_shapes
:??
;sequential_1/fruit_tn1/loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
6sequential_1/fruit_tn1/loop_body/Tensordot_1/transpose	Transpose9sequential_1/fruit_tn1/loop_body/ReadVariableOp_2:value:0Dsequential_1/fruit_tn1/loop_body/Tensordot_1/transpose/perm:output:0*
T0*"
_output_shapes
:?
:sequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
4sequential_1/fruit_tn1/loop_body/Tensordot_1/ReshapeReshape:sequential_1/fruit_tn1/loop_body/Tensordot_1/transpose:y:0Csequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:?
<sequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
6sequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1Reshape3sequential_1/fruit_tn1/loop_body/Tensordot:output:0Esequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?(?
3sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMulMatMul=sequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape:output:0?sequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	?(?
2sequential_1/fruit_tn1/loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"               ?
,sequential_1/fruit_tn1/loop_body/Tensordot_1Reshape=sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul:product:0;sequential_1/fruit_tn1/loop_body/Tensordot_1/shape:output:0*
T0*+
_output_shapes
:??
:sequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
4sequential_1/fruit_tn1/loop_body/Tensordot_2/ReshapeReshape9sequential_1/fruit_tn1/loop_body/ReadVariableOp_1:value:0Csequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes
:	?2?
;sequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
6sequential_1/fruit_tn1/loop_body/Tensordot_2/transpose	Transpose5sequential_1/fruit_tn1/loop_body/Tensordot_1:output:0Dsequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/perm:output:0*
T0*+
_output_shapes
:??
<sequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
6sequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1Reshape:sequential_1/fruit_tn1/loop_body/Tensordot_2/transpose:y:0Esequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2?
3sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMulMatMul=sequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape:output:0?sequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1:output:0*
T0*
_output_shapes

:?
2sequential_1/fruit_tn1/loop_body/Tensordot_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
,sequential_1/fruit_tn1/loop_body/Tensordot_2Reshape=sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul:product:0;sequential_1/fruit_tn1/loop_body/Tensordot_2/shape:output:0*
T0*"
_output_shapes
:?
/sequential_1/fruit_tn1/loop_body/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
*sequential_1/fruit_tn1/loop_body/transpose	Transpose5sequential_1/fruit_tn1/loop_body/Tensordot_2:output:08sequential_1/fruit_tn1/loop_body/transpose/perm:output:0*
T0*"
_output_shapes
:?
3sequential_1/fruit_tn1/loop_body/add/ReadVariableOpReadVariableOp<sequential_1_fruit_tn1_loop_body_add_readvariableop_resource*"
_output_shapes
:*
dtype0?
$sequential_1/fruit_tn1/loop_body/addAddV2.sequential_1/fruit_tn1/loop_body/transpose:y:0;sequential_1/fruit_tn1/loop_body/add/ReadVariableOp:value:0*
T0*"
_output_shapes
:s
)sequential_1/fruit_tn1/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
#sequential_1/fruit_tn1/pfor/ReshapeReshape#sequential_1/fruit_tn1/Max:output:02sequential_1/fruit_tn1/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:i
'sequential_1/fruit_tn1/pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : i
'sequential_1/fruit_tn1/pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
!sequential_1/fruit_tn1/pfor/rangeRange0sequential_1/fruit_tn1/pfor/range/start:output:0#sequential_1/fruit_tn1/Max:output:00sequential_1/fruit_tn1/pfor/range/delta:output:0*#
_output_shapes
:?????????u
3sequential_1/fruit_tn1/loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : v
4sequential_1/fruit_tn1/loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
2sequential_1/fruit_tn1/loop_body/SelectV2/pfor/addAddV2<sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Rank:output:0=sequential_1/fruit_tn1/loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: w
5sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :w
5sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : x
6sequential_1/fruit_tn1/loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
4sequential_1/fruit_tn1/loop_body/SelectV2/pfor/add_1AddV2>sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Rank_2:output:0?sequential_1/fruit_tn1/loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: ?
6sequential_1/fruit_tn1/loop_body/SelectV2/pfor/MaximumMaximum>sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Rank_1:output:06sequential_1/fruit_tn1/loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: ?
8sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Maximum_1Maximum8sequential_1/fruit_tn1/loop_body/SelectV2/pfor/add_1:z:0:sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: ?
4sequential_1/fruit_tn1/loop_body/SelectV2/pfor/ShapeShape*sequential_1/fruit_tn1/pfor/range:output:0*
T0*
_output_shapes
:?
2sequential_1/fruit_tn1/loop_body/SelectV2/pfor/subSub<sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Maximum_1:z:0>sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: ?
<sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
6sequential_1/fruit_tn1/loop_body/SelectV2/pfor/ReshapeReshape6sequential_1/fruit_tn1/loop_body/SelectV2/pfor/sub:z:0Esequential_1/fruit_tn1/loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:?
9sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:?
3sequential_1/fruit_tn1/loop_body/SelectV2/pfor/TileTileBsequential_1/fruit_tn1/loop_body/SelectV2/pfor/Tile/input:output:0?sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: ?
Bsequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Dsequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Dsequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
<sequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_sliceStridedSlice=sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Shape:output:0Ksequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack:output:0Msequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_1:output:0Msequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Dsequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Fsequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Fsequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
>sequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1StridedSlice=sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Shape:output:0Msequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stack:output:0Osequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:0Osequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask|
:sequential_1/fruit_tn1/loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
5sequential_1/fruit_tn1/loop_body/SelectV2/pfor/concatConcatV2Esequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice:output:0<sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Tile:output:0Gsequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1:output:0Csequential_1/fruit_tn1/loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
8sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Reshape_1Reshape*sequential_1/fruit_tn1/pfor/range:output:0>sequential_1/fruit_tn1/loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:??????????
7sequential_1/fruit_tn1/loop_body/SelectV2/pfor/SelectV2SelectV2,sequential_1/fruit_tn1/loop_body/Greater:z:0Asequential_1/fruit_tn1/loop_body/SelectV2/pfor/Reshape_1:output:04sequential_1/fruit_tn1/loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:?????????~
<sequential_1/fruit_tn1/loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
7sequential_1/fruit_tn1/loop_body/GatherV2/pfor/GatherV2GatherV2-sequential_1/max_pooling2d_1/MaxPool:output:0@sequential_1/fruit_tn1/loop_body/SelectV2/pfor/SelectV2:output:0Esequential_1/fruit_tn1/loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*0
_output_shapes
:???????????
?sequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
=sequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/addAddV2Bsequential_1/fruit_tn1/loop_body/Tensordot/transpose/perm:output:0Hsequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:?
Isequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: ?
Esequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
@sequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/concatConcatV2Rsequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/concat/values_0:output:0Asequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/add:z:0Nsequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Csequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/Transpose	Transpose@sequential_1/fruit_tn1/loop_body/GatherV2/pfor/GatherV2:output:0Isequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*0
_output_shapes
:???????????
Csequential_1/fruit_tn1/loop_body/Tensordot/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>sequential_1/fruit_tn1/loop_body/Tensordot/Reshape/pfor/concatConcatV2,sequential_1/fruit_tn1/pfor/Reshape:output:0Asequential_1/fruit_tn1/loop_body/Tensordot/Reshape/shape:output:0Lsequential_1/fruit_tn1/loop_body/Tensordot/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
?sequential_1/fruit_tn1/loop_body/Tensordot/Reshape/pfor/ReshapeReshapeGsequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/Transpose:y:0Gsequential_1/fruit_tn1/loop_body/Tensordot/Reshape/pfor/concat:output:0*
T0*,
_output_shapes
:???????????
<sequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/ShapeShapeHsequential_1/fruit_tn1/loop_body/Tensordot/Reshape/pfor/Reshape:output:0*
T0*
_output_shapes
:?
Fsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
<sequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/splitSplitOsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/split/split_dim:output:0Esequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Shape:output:0*
T0*&
_output_shapes
:::*
	num_split?
Dsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Fsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
>sequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/ReshapeReshapeEsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/split:output:0Osequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1:output:0*
T0*
_output_shapes
: ?
Fsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Hsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
@sequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1ReshapeEsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/split:output:1Qsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ?
Fsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Hsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
@sequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2ReshapeEsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/split:output:2Qsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ?
:sequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/mulMulGsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape:output:0Isequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: ?
Fsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack>sequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/mul:z:0Isequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:?
@sequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3ReshapeHsequential_1/fruit_tn1/loop_body/Tensordot/Reshape/pfor/Reshape:output:0Osequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:???????????????????
=sequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/MatMulMatMulIsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0=sequential_1/fruit_tn1/loop_body/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:??????????
Hsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
??????????
Fsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePackGsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape:output:0Isequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0Qsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:?
@sequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4ReshapeGsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/MatMul:product:0Osequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*,
_output_shapes
:??????????}
;sequential_1/fruit_tn1/loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6sequential_1/fruit_tn1/loop_body/Tensordot/pfor/concatConcatV2,sequential_1/fruit_tn1/pfor/Reshape:output:09sequential_1/fruit_tn1/loop_body/Tensordot/shape:output:0Dsequential_1/fruit_tn1/loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
7sequential_1/fruit_tn1/loop_body/Tensordot/pfor/ReshapeReshapeIsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4:output:0?sequential_1/fruit_tn1/loop_body/Tensordot/pfor/concat:output:0*
T0*4
_output_shapes"
 :???????????
Gsequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Bsequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2,sequential_1/fruit_tn1/pfor/Reshape:output:0Esequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1/shape:output:0Psequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Csequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape@sequential_1/fruit_tn1/loop_body/Tensordot/pfor/Reshape:output:0Ksequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:??????????(?
>sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/ShapeShapeLsequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:?
Lsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Nsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Nsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Fsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSliceGsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Usequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Nsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSliceGsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/MinimumMinimumOsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0Qsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: ?
@sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :?
>sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/EqualEqualDsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Minimum:z:0Isequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: ?
Asequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          ?
Asequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          ?
?sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/SelectSelectBsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Equal:z:0Jsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/t:output:0Jsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:?
Nsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSliceGsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Nsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Psequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSliceGsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Nsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSliceGsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
>sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/stackPackQsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0Qsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0Qsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
Bsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose	TransposeLsequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0Hsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'????????????????????????????
@sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/ReshapeReshapeFsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose:y:0Gsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:??????????(?
@sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape_1ShapeIsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:?
Hsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
>sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/splitSplitQsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split/split_dim:output:0Isequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split?
Hsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Jsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Bsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1ReshapeGsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split:output:0Ssequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ?
Hsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Jsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Bsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2ReshapeGsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split:output:1Ssequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ?
Hsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Jsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Bsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3ReshapeGsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split:output:2Ssequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ?
<sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/mulMulKsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0Ksequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ?
Hsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePackKsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:0@sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:?
Bsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4ReshapeIsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0Qsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:???????????????????
?sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul=sequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape:output:0Ksequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:??????????
Jsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
??????????
Hsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePackSsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:0Ksequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0Ksequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:?
Bsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5ReshapeIsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0Qsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'????????????????????????????
Isequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
Dsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1	TransposeKsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0Rsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????(
=sequential_1/fruit_tn1/loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
8sequential_1/fruit_tn1/loop_body/Tensordot_1/pfor/concatConcatV2,sequential_1/fruit_tn1/pfor/Reshape:output:0;sequential_1/fruit_tn1/loop_body/Tensordot_1/shape:output:0Fsequential_1/fruit_tn1/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
9sequential_1/fruit_tn1/loop_body/Tensordot_1/pfor/ReshapeReshapeHsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:0Asequential_1/fruit_tn1/loop_body/Tensordot_1/pfor/concat:output:0*
T0*8
_output_shapes&
$:"???????????
Asequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
?sequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/addAddV2Dsequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/perm:output:0Jsequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:?
Ksequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: ?
Gsequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Bsequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concatConcatV2Tsequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/values_0:output:0Csequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/add:z:0Psequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Esequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/Transpose	TransposeBsequential_1/fruit_tn1/loop_body/Tensordot_1/pfor/Reshape:output:0Ksequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat:output:0*
T0*8
_output_shapes&
$:"???????????
Gsequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Bsequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concatConcatV2,sequential_1/fruit_tn1/pfor/Reshape:output:0Esequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1/shape:output:0Psequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Csequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/ReshapeReshapeIsequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/Transpose:y:0Ksequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:??????????2?
>sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/ShapeShapeLsequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:?
Lsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Nsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Nsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Fsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_sliceStridedSliceGsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Usequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Nsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1StridedSliceGsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/MinimumMinimumOsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice:output:0Qsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: ?
@sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :?
>sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/EqualEqualDsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Minimum:z:0Isequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: ?
Asequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          ?
Asequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          ?
?sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/SelectSelectBsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Equal:z:0Jsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/t:output:0Jsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:?
Nsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2StridedSliceGsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Nsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Psequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3StridedSliceGsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Nsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4StridedSliceGsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
>sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/stackPackQsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2:output:0Qsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3:output:0Qsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
Bsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose	TransposeLsequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0Hsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'????????????????????????????
@sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/ReshapeReshapeFsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose:y:0Gsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:?2??????????
@sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape_1ShapeIsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:?
Hsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
>sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/splitSplitQsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split/split_dim:output:0Isequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split?
Hsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Jsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Bsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1ReshapeGsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split:output:0Ssequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ?
Hsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Jsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Bsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2ReshapeGsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split:output:1Ssequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ?
Hsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Jsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Bsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3ReshapeGsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split:output:2Ssequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ?
<sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/mulMulKsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0Ksequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ?
Hsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shapePackKsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1:output:0@sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:?
Bsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4ReshapeIsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0Qsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:???????????????????
?sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/MatMulMatMul=sequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape:output:0Ksequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:??????????
Jsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
??????????
Hsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shapePackSsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0:output:0Ksequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0Ksequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:?
Bsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5ReshapeIsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/MatMul:product:0Qsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'????????????????????????????
Isequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
Dsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1	TransposeKsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5:output:0Rsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
=sequential_1/fruit_tn1/loop_body/Tensordot_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
8sequential_1/fruit_tn1/loop_body/Tensordot_2/pfor/concatConcatV2,sequential_1/fruit_tn1/pfor/Reshape:output:0;sequential_1/fruit_tn1/loop_body/Tensordot_2/shape:output:0Fsequential_1/fruit_tn1/loop_body/Tensordot_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
9sequential_1/fruit_tn1/loop_body/Tensordot_2/pfor/ReshapeReshapeHsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1:y:0Asequential_1/fruit_tn1/loop_body/Tensordot_2/pfor/concat:output:0*
T0*/
_output_shapes
:?????????w
5sequential_1/fruit_tn1/loop_body/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
3sequential_1/fruit_tn1/loop_body/transpose/pfor/addAddV28sequential_1/fruit_tn1/loop_body/transpose/perm:output:0>sequential_1/fruit_tn1/loop_body/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:?
?sequential_1/fruit_tn1/loop_body/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: }
;sequential_1/fruit_tn1/loop_body/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6sequential_1/fruit_tn1/loop_body/transpose/pfor/concatConcatV2Hsequential_1/fruit_tn1/loop_body/transpose/pfor/concat/values_0:output:07sequential_1/fruit_tn1/loop_body/transpose/pfor/add:z:0Dsequential_1/fruit_tn1/loop_body/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
9sequential_1/fruit_tn1/loop_body/transpose/pfor/Transpose	TransposeBsequential_1/fruit_tn1/loop_body/Tensordot_2/pfor/Reshape:output:0?sequential_1/fruit_tn1/loop_body/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:?????????p
.sequential_1/fruit_tn1/loop_body/add/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :r
0sequential_1/fruit_tn1/loop_body/add/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :q
/sequential_1/fruit_tn1/loop_body/add/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
-sequential_1/fruit_tn1/loop_body/add/pfor/addAddV29sequential_1/fruit_tn1/loop_body/add/pfor/Rank_1:output:08sequential_1/fruit_tn1/loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: ?
1sequential_1/fruit_tn1/loop_body/add/pfor/MaximumMaximum1sequential_1/fruit_tn1/loop_body/add/pfor/add:z:07sequential_1/fruit_tn1/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: ?
/sequential_1/fruit_tn1/loop_body/add/pfor/ShapeShape=sequential_1/fruit_tn1/loop_body/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:?
-sequential_1/fruit_tn1/loop_body/add/pfor/subSub5sequential_1/fruit_tn1/loop_body/add/pfor/Maximum:z:07sequential_1/fruit_tn1/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: ?
7sequential_1/fruit_tn1/loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
1sequential_1/fruit_tn1/loop_body/add/pfor/ReshapeReshape1sequential_1/fruit_tn1/loop_body/add/pfor/sub:z:0@sequential_1/fruit_tn1/loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:~
4sequential_1/fruit_tn1/loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:?
.sequential_1/fruit_tn1/loop_body/add/pfor/TileTile=sequential_1/fruit_tn1/loop_body/add/pfor/Tile/input:output:0:sequential_1/fruit_tn1/loop_body/add/pfor/Reshape:output:0*
T0*
_output_shapes
: ?
=sequential_1/fruit_tn1/loop_body/add/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?sequential_1/fruit_tn1/loop_body/add/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?sequential_1/fruit_tn1/loop_body/add/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
7sequential_1/fruit_tn1/loop_body/add/pfor/strided_sliceStridedSlice8sequential_1/fruit_tn1/loop_body/add/pfor/Shape:output:0Fsequential_1/fruit_tn1/loop_body/add/pfor/strided_slice/stack:output:0Hsequential_1/fruit_tn1/loop_body/add/pfor/strided_slice/stack_1:output:0Hsequential_1/fruit_tn1/loop_body/add/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
?sequential_1/fruit_tn1/loop_body/add/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Asequential_1/fruit_tn1/loop_body/add/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Asequential_1/fruit_tn1/loop_body/add/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9sequential_1/fruit_tn1/loop_body/add/pfor/strided_slice_1StridedSlice8sequential_1/fruit_tn1/loop_body/add/pfor/Shape:output:0Hsequential_1/fruit_tn1/loop_body/add/pfor/strided_slice_1/stack:output:0Jsequential_1/fruit_tn1/loop_body/add/pfor/strided_slice_1/stack_1:output:0Jsequential_1/fruit_tn1/loop_body/add/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskw
5sequential_1/fruit_tn1/loop_body/add/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0sequential_1/fruit_tn1/loop_body/add/pfor/concatConcatV2@sequential_1/fruit_tn1/loop_body/add/pfor/strided_slice:output:07sequential_1/fruit_tn1/loop_body/add/pfor/Tile:output:0Bsequential_1/fruit_tn1/loop_body/add/pfor/strided_slice_1:output:0>sequential_1/fruit_tn1/loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
3sequential_1/fruit_tn1/loop_body/add/pfor/Reshape_1Reshape=sequential_1/fruit_tn1/loop_body/transpose/pfor/Transpose:y:09sequential_1/fruit_tn1/loop_body/add/pfor/concat:output:0*
T0*/
_output_shapes
:??????????
/sequential_1/fruit_tn1/loop_body/add/pfor/AddV2AddV2<sequential_1/fruit_tn1/loop_body/add/pfor/Reshape_1:output:0;sequential_1/fruit_tn1/loop_body/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????u
$sequential_1/fruit_tn1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
sequential_1/fruit_tn1/ReshapeReshape3sequential_1/fruit_tn1/loop_body/add/pfor/AddV2:z:0-sequential_1/fruit_tn1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????@~
sequential_1/fruit_tn1/ReluRelu'sequential_1/fruit_tn1/Reshape:output:0*
T0*'
_output_shapes
:?????????@?
1sequential_1/fruit_tn1/ActivityRegularizer/SquareSquare)sequential_1/fruit_tn1/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
0sequential_1/fruit_tn1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
.sequential_1/fruit_tn1/ActivityRegularizer/SumSum5sequential_1/fruit_tn1/ActivityRegularizer/Square:y:09sequential_1/fruit_tn1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: u
0sequential_1/fruit_tn1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
.sequential_1/fruit_tn1/ActivityRegularizer/mulMul9sequential_1/fruit_tn1/ActivityRegularizer/mul/x:output:07sequential_1/fruit_tn1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: ?
0sequential_1/fruit_tn1/ActivityRegularizer/ShapeShape)sequential_1/fruit_tn1/Relu:activations:0*
T0*
_output_shapes
:?
>sequential_1/fruit_tn1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
@sequential_1/fruit_tn1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@sequential_1/fruit_tn1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_1/fruit_tn1/ActivityRegularizer/strided_sliceStridedSlice9sequential_1/fruit_tn1/ActivityRegularizer/Shape:output:0Gsequential_1/fruit_tn1/ActivityRegularizer/strided_slice/stack:output:0Isequential_1/fruit_tn1/ActivityRegularizer/strided_slice/stack_1:output:0Isequential_1/fruit_tn1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
/sequential_1/fruit_tn1/ActivityRegularizer/CastCastAsequential_1/fruit_tn1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
2sequential_1/fruit_tn1/ActivityRegularizer/truedivRealDiv2sequential_1/fruit_tn1/ActivityRegularizer/mul:z:03sequential_1/fruit_tn1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
sequential_1/dropout1/IdentityIdentity)sequential_1/fruit_tn1/Relu:activations:0*
T0*'
_output_shapes
:?????????@s
sequential_1/fruit_tn2/ShapeShape'sequential_1/dropout1/Identity:output:0*
T0*
_output_shapes
:t
*sequential_1/fruit_tn2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_1/fruit_tn2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_1/fruit_tn2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$sequential_1/fruit_tn2/strided_sliceStridedSlice%sequential_1/fruit_tn2/Shape:output:03sequential_1/fruit_tn2/strided_slice/stack:output:05sequential_1/fruit_tn2/strided_slice/stack_1:output:05sequential_1/fruit_tn2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"sequential_1/fruit_tn2/Rank/packedPack-sequential_1/fruit_tn2/strided_slice:output:0*
N*
T0*
_output_shapes
:]
sequential_1/fruit_tn2/RankConst*
_output_shapes
: *
dtype0*
value	B :d
"sequential_1/fruit_tn2/range/startConst*
_output_shapes
: *
dtype0*
value	B : d
"sequential_1/fruit_tn2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
sequential_1/fruit_tn2/rangeRange+sequential_1/fruit_tn2/range/start:output:0$sequential_1/fruit_tn2/Rank:output:0+sequential_1/fruit_tn2/range/delta:output:0*
_output_shapes
:?
 sequential_1/fruit_tn2/Max/inputPack-sequential_1/fruit_tn2/strided_slice:output:0*
N*
T0*
_output_shapes
:?
sequential_1/fruit_tn2/MaxMax)sequential_1/fruit_tn2/Max/input:output:0%sequential_1/fruit_tn2/range:output:0*
T0*
_output_shapes
: 
=sequential_1/fruit_tn2/loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : ?
7sequential_1/fruit_tn2/loop_body/PlaceholderWithDefaultPlaceholderWithDefaultFsequential_1/fruit_tn2/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: }
&sequential_1/fruit_tn2/loop_body/ShapeShape'sequential_1/dropout1/Identity:output:0*
T0*
_output_shapes
:~
4sequential_1/fruit_tn2/loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6sequential_1/fruit_tn2/loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6sequential_1/fruit_tn2/loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.sequential_1/fruit_tn2/loop_body/strided_sliceStridedSlice/sequential_1/fruit_tn2/loop_body/Shape:output:0=sequential_1/fruit_tn2/loop_body/strided_slice/stack:output:0?sequential_1/fruit_tn2/loop_body/strided_slice/stack_1:output:0?sequential_1/fruit_tn2/loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*sequential_1/fruit_tn2/loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :?
(sequential_1/fruit_tn2/loop_body/GreaterGreater7sequential_1/fruit_tn2/loop_body/strided_slice:output:03sequential_1/fruit_tn2/loop_body/Greater/y:output:0*
T0*
_output_shapes
: m
+sequential_1/fruit_tn2/loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_1/fruit_tn2/loop_body/SelectV2SelectV2,sequential_1/fruit_tn2/loop_body/Greater:z:0@sequential_1/fruit_tn2/loop_body/PlaceholderWithDefault:output:04sequential_1/fruit_tn2/loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: p
.sequential_1/fruit_tn2/loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_1/fruit_tn2/loop_body/GatherV2GatherV2'sequential_1/dropout1/Identity:output:02sequential_1/fruit_tn2/loop_body/SelectV2:output:07sequential_1/fruit_tn2/loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:@?
.sequential_1/fruit_tn2/loop_body/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
(sequential_1/fruit_tn2/loop_body/ReshapeReshape2sequential_1/fruit_tn2/loop_body/GatherV2:output:07sequential_1/fruit_tn2/loop_body/Reshape/shape:output:0*
T0*"
_output_shapes
:?
/sequential_1/fruit_tn2/loop_body/ReadVariableOpReadVariableOp8sequential_1_fruit_tn2_loop_body_readvariableop_resource*
_output_shapes

:*
dtype0?
1sequential_1/fruit_tn2/loop_body/ReadVariableOp_1ReadVariableOp:sequential_1_fruit_tn2_loop_body_readvariableop_1_resource*'
_output_shapes
:?*
dtype0?
1sequential_1/fruit_tn2/loop_body/ReadVariableOp_2ReadVariableOp:sequential_1_fruit_tn2_loop_body_readvariableop_2_resource*
_output_shapes

:*
dtype0?
9sequential_1/fruit_tn2/loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
4sequential_1/fruit_tn2/loop_body/Tensordot/transpose	Transpose1sequential_1/fruit_tn2/loop_body/Reshape:output:0Bsequential_1/fruit_tn2/loop_body/Tensordot/transpose/perm:output:0*
T0*"
_output_shapes
:?
8sequential_1/fruit_tn2/loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
2sequential_1/fruit_tn2/loop_body/Tensordot/ReshapeReshape8sequential_1/fruit_tn2/loop_body/Tensordot/transpose:y:0Asequential_1/fruit_tn2/loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:?
1sequential_1/fruit_tn2/loop_body/Tensordot/MatMulMatMul;sequential_1/fruit_tn2/loop_body/Tensordot/Reshape:output:07sequential_1/fruit_tn2/loop_body/ReadVariableOp:value:0*
T0*
_output_shapes

:?
0sequential_1/fruit_tn2/loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
*sequential_1/fruit_tn2/loop_body/TensordotReshape;sequential_1/fruit_tn2/loop_body/Tensordot/MatMul:product:09sequential_1/fruit_tn2/loop_body/Tensordot/shape:output:0*
T0*"
_output_shapes
:?
;sequential_1/fruit_tn2/loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
6sequential_1/fruit_tn2/loop_body/Tensordot_1/transpose	Transpose9sequential_1/fruit_tn2/loop_body/ReadVariableOp_1:value:0Dsequential_1/fruit_tn2/loop_body/Tensordot_1/transpose/perm:output:0*
T0*'
_output_shapes
:??
:sequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"     ?
4sequential_1/fruit_tn2/loop_body/Tensordot_1/ReshapeReshape:sequential_1/fruit_tn2/loop_body/Tensordot_1/transpose:y:0Csequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes
:	??
=sequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
8sequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1	Transpose3sequential_1/fruit_tn2/loop_body/Tensordot:output:0Fsequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/perm:output:0*
T0*"
_output_shapes
:?
<sequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
6sequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1Reshape<sequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1:y:0Esequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:?
3sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMulMatMul=sequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape:output:0?sequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	??
2sequential_1/fruit_tn2/loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"?         ?
,sequential_1/fruit_tn2/loop_body/Tensordot_1Reshape=sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul:product:0;sequential_1/fruit_tn2/loop_body/Tensordot_1/shape:output:0*
T0*#
_output_shapes
:??
:sequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
4sequential_1/fruit_tn2/loop_body/Tensordot_2/ReshapeReshape9sequential_1/fruit_tn2/loop_body/ReadVariableOp_2:value:0Csequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes

:?
;sequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
6sequential_1/fruit_tn2/loop_body/Tensordot_2/transpose	Transpose5sequential_1/fruit_tn2/loop_body/Tensordot_1:output:0Dsequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/perm:output:0*
T0*#
_output_shapes
:??
<sequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   ?
6sequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1Reshape:sequential_1/fruit_tn2/loop_body/Tensordot_2/transpose:y:0Esequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	??
3sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMulMatMul=sequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape:output:0?sequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1:output:0*
T0*
_output_shapes
:	?}
2sequential_1/fruit_tn2/loop_body/Tensordot_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:??
,sequential_1/fruit_tn2/loop_body/Tensordot_2Reshape=sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul:product:0;sequential_1/fruit_tn2/loop_body/Tensordot_2/shape:output:0*
T0*
_output_shapes	
:??
3sequential_1/fruit_tn2/loop_body/add/ReadVariableOpReadVariableOp<sequential_1_fruit_tn2_loop_body_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$sequential_1/fruit_tn2/loop_body/addAddV25sequential_1/fruit_tn2/loop_body/Tensordot_2:output:0;sequential_1/fruit_tn2/loop_body/add/ReadVariableOp:value:0*
T0*
_output_shapes	
:?s
)sequential_1/fruit_tn2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
#sequential_1/fruit_tn2/pfor/ReshapeReshape#sequential_1/fruit_tn2/Max:output:02sequential_1/fruit_tn2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:i
'sequential_1/fruit_tn2/pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : i
'sequential_1/fruit_tn2/pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
!sequential_1/fruit_tn2/pfor/rangeRange0sequential_1/fruit_tn2/pfor/range/start:output:0#sequential_1/fruit_tn2/Max:output:00sequential_1/fruit_tn2/pfor/range/delta:output:0*#
_output_shapes
:?????????u
3sequential_1/fruit_tn2/loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : v
4sequential_1/fruit_tn2/loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
2sequential_1/fruit_tn2/loop_body/SelectV2/pfor/addAddV2<sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Rank:output:0=sequential_1/fruit_tn2/loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: w
5sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :w
5sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : x
6sequential_1/fruit_tn2/loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
4sequential_1/fruit_tn2/loop_body/SelectV2/pfor/add_1AddV2>sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Rank_2:output:0?sequential_1/fruit_tn2/loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: ?
6sequential_1/fruit_tn2/loop_body/SelectV2/pfor/MaximumMaximum>sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Rank_1:output:06sequential_1/fruit_tn2/loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: ?
8sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Maximum_1Maximum8sequential_1/fruit_tn2/loop_body/SelectV2/pfor/add_1:z:0:sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: ?
4sequential_1/fruit_tn2/loop_body/SelectV2/pfor/ShapeShape*sequential_1/fruit_tn2/pfor/range:output:0*
T0*
_output_shapes
:?
2sequential_1/fruit_tn2/loop_body/SelectV2/pfor/subSub<sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Maximum_1:z:0>sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: ?
<sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
6sequential_1/fruit_tn2/loop_body/SelectV2/pfor/ReshapeReshape6sequential_1/fruit_tn2/loop_body/SelectV2/pfor/sub:z:0Esequential_1/fruit_tn2/loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:?
9sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:?
3sequential_1/fruit_tn2/loop_body/SelectV2/pfor/TileTileBsequential_1/fruit_tn2/loop_body/SelectV2/pfor/Tile/input:output:0?sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: ?
Bsequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Dsequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Dsequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
<sequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_sliceStridedSlice=sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Shape:output:0Ksequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack:output:0Msequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_1:output:0Msequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Dsequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Fsequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Fsequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
>sequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1StridedSlice=sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Shape:output:0Msequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stack:output:0Osequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:0Osequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask|
:sequential_1/fruit_tn2/loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
5sequential_1/fruit_tn2/loop_body/SelectV2/pfor/concatConcatV2Esequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice:output:0<sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Tile:output:0Gsequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1:output:0Csequential_1/fruit_tn2/loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
8sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Reshape_1Reshape*sequential_1/fruit_tn2/pfor/range:output:0>sequential_1/fruit_tn2/loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:??????????
7sequential_1/fruit_tn2/loop_body/SelectV2/pfor/SelectV2SelectV2,sequential_1/fruit_tn2/loop_body/Greater:z:0Asequential_1/fruit_tn2/loop_body/SelectV2/pfor/Reshape_1:output:04sequential_1/fruit_tn2/loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:?????????~
<sequential_1/fruit_tn2/loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
7sequential_1/fruit_tn2/loop_body/GatherV2/pfor/GatherV2GatherV2'sequential_1/dropout1/Identity:output:0@sequential_1/fruit_tn2/loop_body/SelectV2/pfor/SelectV2:output:0Esequential_1/fruit_tn2/loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:?????????@{
9sequential_1/fruit_tn2/loop_body/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4sequential_1/fruit_tn2/loop_body/Reshape/pfor/concatConcatV2,sequential_1/fruit_tn2/pfor/Reshape:output:07sequential_1/fruit_tn2/loop_body/Reshape/shape:output:0Bsequential_1/fruit_tn2/loop_body/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
5sequential_1/fruit_tn2/loop_body/Reshape/pfor/ReshapeReshape@sequential_1/fruit_tn2/loop_body/GatherV2/pfor/GatherV2:output:0=sequential_1/fruit_tn2/loop_body/Reshape/pfor/concat:output:0*
T0*/
_output_shapes
:??????????
?sequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
=sequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/addAddV2Bsequential_1/fruit_tn2/loop_body/Tensordot/transpose/perm:output:0Hsequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:?
Isequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: ?
Esequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
@sequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/concatConcatV2Rsequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/concat/values_0:output:0Asequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/add:z:0Nsequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Csequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/Transpose	Transpose>sequential_1/fruit_tn2/loop_body/Reshape/pfor/Reshape:output:0Isequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:??????????
Csequential_1/fruit_tn2/loop_body/Tensordot/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>sequential_1/fruit_tn2/loop_body/Tensordot/Reshape/pfor/concatConcatV2,sequential_1/fruit_tn2/pfor/Reshape:output:0Asequential_1/fruit_tn2/loop_body/Tensordot/Reshape/shape:output:0Lsequential_1/fruit_tn2/loop_body/Tensordot/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
?sequential_1/fruit_tn2/loop_body/Tensordot/Reshape/pfor/ReshapeReshapeGsequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/Transpose:y:0Gsequential_1/fruit_tn2/loop_body/Tensordot/Reshape/pfor/concat:output:0*
T0*+
_output_shapes
:??????????
<sequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/ShapeShapeHsequential_1/fruit_tn2/loop_body/Tensordot/Reshape/pfor/Reshape:output:0*
T0*
_output_shapes
:?
Fsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
<sequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/splitSplitOsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/split/split_dim:output:0Esequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Shape:output:0*
T0*&
_output_shapes
:::*
	num_split?
Dsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Fsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
>sequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/ReshapeReshapeEsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/split:output:0Osequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1:output:0*
T0*
_output_shapes
: ?
Fsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Hsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
@sequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1ReshapeEsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/split:output:1Qsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ?
Fsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Hsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
@sequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2ReshapeEsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/split:output:2Qsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ?
:sequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/mulMulGsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape:output:0Isequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: ?
Fsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack>sequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/mul:z:0Isequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:?
@sequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3ReshapeHsequential_1/fruit_tn2/loop_body/Tensordot/Reshape/pfor/Reshape:output:0Osequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:???????????????????
=sequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/MatMulMatMulIsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3:output:07sequential_1/fruit_tn2/loop_body/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Hsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
??????????
Fsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePackGsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape:output:0Isequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0Qsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:?
@sequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4ReshapeGsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/MatMul:product:0Osequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*+
_output_shapes
:?????????}
;sequential_1/fruit_tn2/loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6sequential_1/fruit_tn2/loop_body/Tensordot/pfor/concatConcatV2,sequential_1/fruit_tn2/pfor/Reshape:output:09sequential_1/fruit_tn2/loop_body/Tensordot/shape:output:0Dsequential_1/fruit_tn2/loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
7sequential_1/fruit_tn2/loop_body/Tensordot/pfor/ReshapeReshapeIsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4:output:0?sequential_1/fruit_tn2/loop_body/Tensordot/pfor/concat:output:0*
T0*/
_output_shapes
:??????????
Csequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
Asequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/addAddV2Fsequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/perm:output:0Lsequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/add/y:output:0*
T0*
_output_shapes
:?
Msequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: ?
Isequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Dsequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concatConcatV2Vsequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/values_0:output:0Esequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/add:z:0Rsequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Gsequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/Transpose	Transpose@sequential_1/fruit_tn2/loop_body/Tensordot/pfor/Reshape:output:0Msequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat:output:0*
T0*/
_output_shapes
:??????????
Gsequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Bsequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2,sequential_1/fruit_tn2/pfor/Reshape:output:0Esequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1/shape:output:0Psequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Csequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshapeKsequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/Transpose:y:0Ksequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*+
_output_shapes
:??????????
>sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/ShapeShapeLsequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:?
Lsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Nsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Nsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Fsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSliceGsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Usequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Nsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSliceGsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/MinimumMinimumOsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0Qsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: ?
@sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :?
>sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/EqualEqualDsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Minimum:z:0Isequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: ?
Asequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          ?
Asequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          ?
?sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/SelectSelectBsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Equal:z:0Jsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/t:output:0Jsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:?
Nsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSliceGsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Nsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Psequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSliceGsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Nsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSliceGsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
>sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/stackPackQsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0Qsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0Qsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
Bsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose	TransposeLsequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0Hsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'????????????????????????????
@sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/ReshapeReshapeFsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose:y:0Gsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*+
_output_shapes
:??????????
@sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape_1ShapeIsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:?
Hsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
>sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/splitSplitQsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split/split_dim:output:0Isequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split?
Hsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Jsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Bsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1ReshapeGsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split:output:0Ssequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ?
Hsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Jsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Bsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2ReshapeGsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split:output:1Ssequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ?
Hsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Jsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Bsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3ReshapeGsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split:output:2Ssequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ?
<sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/mulMulKsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0Ksequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ?
Hsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePackKsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:0@sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:?
Bsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4ReshapeIsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0Qsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:???????????????????
?sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul=sequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape:output:0Ksequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*(
_output_shapes
:???????????
Jsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
??????????
Hsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePackSsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:0Ksequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0Ksequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:?
Bsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5ReshapeIsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0Qsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'????????????????????????????
Isequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
Dsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1	TransposeKsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0Rsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????
=sequential_1/fruit_tn2/loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
8sequential_1/fruit_tn2/loop_body/Tensordot_1/pfor/concatConcatV2,sequential_1/fruit_tn2/pfor/Reshape:output:0;sequential_1/fruit_tn2/loop_body/Tensordot_1/shape:output:0Fsequential_1/fruit_tn2/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
9sequential_1/fruit_tn2/loop_body/Tensordot_1/pfor/ReshapeReshapeHsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:0Asequential_1/fruit_tn2/loop_body/Tensordot_1/pfor/concat:output:0*
T0*0
_output_shapes
:???????????
Asequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
?sequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/addAddV2Dsequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/perm:output:0Jsequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:?
Ksequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: ?
Gsequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Bsequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concatConcatV2Tsequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/values_0:output:0Csequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/add:z:0Psequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Esequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/Transpose	TransposeBsequential_1/fruit_tn2/loop_body/Tensordot_1/pfor/Reshape:output:0Ksequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat:output:0*
T0*0
_output_shapes
:???????????
Gsequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Bsequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concatConcatV2,sequential_1/fruit_tn2/pfor/Reshape:output:0Esequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1/shape:output:0Psequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Csequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/ReshapeReshapeIsequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/Transpose:y:0Ksequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:???????????
>sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/ShapeShapeLsequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:?
Lsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Nsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Nsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Fsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_sliceStridedSliceGsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Usequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Nsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1StridedSliceGsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/MinimumMinimumOsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice:output:0Qsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: ?
@sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :?
>sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/EqualEqualDsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Minimum:z:0Isequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: ?
Asequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          ?
Asequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          ?
?sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/SelectSelectBsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Equal:z:0Jsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/t:output:0Jsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:?
Nsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2StridedSliceGsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Nsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Psequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3StridedSliceGsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Nsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Psequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4StridedSliceGsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
>sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/stackPackQsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2:output:0Qsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3:output:0Qsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
Bsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose	TransposeLsequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0Hsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'????????????????????????????
@sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/ReshapeReshapeFsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose:y:0Gsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:???????????
@sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape_1ShapeIsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:?
Hsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
>sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/splitSplitQsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split/split_dim:output:0Isequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split?
Hsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Jsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Bsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1ReshapeGsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split:output:0Ssequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ?
Hsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Jsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Bsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2ReshapeGsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split:output:1Ssequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ?
Hsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Jsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Bsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3ReshapeGsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split:output:2Ssequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ?
<sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/mulMulKsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0Ksequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ?
Hsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shapePackKsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1:output:0@sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:?
Bsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4ReshapeIsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0Qsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:???????????????????
?sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/MatMulMatMul=sequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape:output:0Ksequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:??????????
Jsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
??????????
Hsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shapePackSsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0:output:0Ksequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0Ksequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:?
Bsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5ReshapeIsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/MatMul:product:0Qsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'????????????????????????????
Isequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
Dsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1	TransposeKsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5:output:0Rsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????
=sequential_1/fruit_tn2/loop_body/Tensordot_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
8sequential_1/fruit_tn2/loop_body/Tensordot_2/pfor/concatConcatV2,sequential_1/fruit_tn2/pfor/Reshape:output:0;sequential_1/fruit_tn2/loop_body/Tensordot_2/shape:output:0Fsequential_1/fruit_tn2/loop_body/Tensordot_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
9sequential_1/fruit_tn2/loop_body/Tensordot_2/pfor/ReshapeReshapeHsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1:y:0Asequential_1/fruit_tn2/loop_body/Tensordot_2/pfor/concat:output:0*
T0*(
_output_shapes
:??????????p
.sequential_1/fruit_tn2/loop_body/add/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :r
0sequential_1/fruit_tn2/loop_body/add/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :q
/sequential_1/fruit_tn2/loop_body/add/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
-sequential_1/fruit_tn2/loop_body/add/pfor/addAddV29sequential_1/fruit_tn2/loop_body/add/pfor/Rank_1:output:08sequential_1/fruit_tn2/loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: ?
1sequential_1/fruit_tn2/loop_body/add/pfor/MaximumMaximum1sequential_1/fruit_tn2/loop_body/add/pfor/add:z:07sequential_1/fruit_tn2/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: ?
/sequential_1/fruit_tn2/loop_body/add/pfor/ShapeShapeBsequential_1/fruit_tn2/loop_body/Tensordot_2/pfor/Reshape:output:0*
T0*
_output_shapes
:?
-sequential_1/fruit_tn2/loop_body/add/pfor/subSub5sequential_1/fruit_tn2/loop_body/add/pfor/Maximum:z:07sequential_1/fruit_tn2/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: ?
7sequential_1/fruit_tn2/loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
1sequential_1/fruit_tn2/loop_body/add/pfor/ReshapeReshape1sequential_1/fruit_tn2/loop_body/add/pfor/sub:z:0@sequential_1/fruit_tn2/loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:~
4sequential_1/fruit_tn2/loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:?
.sequential_1/fruit_tn2/loop_body/add/pfor/TileTile=sequential_1/fruit_tn2/loop_body/add/pfor/Tile/input:output:0:sequential_1/fruit_tn2/loop_body/add/pfor/Reshape:output:0*
T0*
_output_shapes
: ?
=sequential_1/fruit_tn2/loop_body/add/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?sequential_1/fruit_tn2/loop_body/add/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?sequential_1/fruit_tn2/loop_body/add/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
7sequential_1/fruit_tn2/loop_body/add/pfor/strided_sliceStridedSlice8sequential_1/fruit_tn2/loop_body/add/pfor/Shape:output:0Fsequential_1/fruit_tn2/loop_body/add/pfor/strided_slice/stack:output:0Hsequential_1/fruit_tn2/loop_body/add/pfor/strided_slice/stack_1:output:0Hsequential_1/fruit_tn2/loop_body/add/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
?sequential_1/fruit_tn2/loop_body/add/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Asequential_1/fruit_tn2/loop_body/add/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Asequential_1/fruit_tn2/loop_body/add/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9sequential_1/fruit_tn2/loop_body/add/pfor/strided_slice_1StridedSlice8sequential_1/fruit_tn2/loop_body/add/pfor/Shape:output:0Hsequential_1/fruit_tn2/loop_body/add/pfor/strided_slice_1/stack:output:0Jsequential_1/fruit_tn2/loop_body/add/pfor/strided_slice_1/stack_1:output:0Jsequential_1/fruit_tn2/loop_body/add/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskw
5sequential_1/fruit_tn2/loop_body/add/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0sequential_1/fruit_tn2/loop_body/add/pfor/concatConcatV2@sequential_1/fruit_tn2/loop_body/add/pfor/strided_slice:output:07sequential_1/fruit_tn2/loop_body/add/pfor/Tile:output:0Bsequential_1/fruit_tn2/loop_body/add/pfor/strided_slice_1:output:0>sequential_1/fruit_tn2/loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
3sequential_1/fruit_tn2/loop_body/add/pfor/Reshape_1ReshapeBsequential_1/fruit_tn2/loop_body/Tensordot_2/pfor/Reshape:output:09sequential_1/fruit_tn2/loop_body/add/pfor/concat:output:0*
T0*(
_output_shapes
:???????????
/sequential_1/fruit_tn2/loop_body/add/pfor/AddV2AddV2<sequential_1/fruit_tn2/loop_body/add/pfor/Reshape_1:output:0;sequential_1/fruit_tn2/loop_body/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????u
$sequential_1/fruit_tn2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
sequential_1/fruit_tn2/ReshapeReshape3sequential_1/fruit_tn2/loop_body/add/pfor/AddV2:z:0-sequential_1/fruit_tn2/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
sequential_1/fruit_tn2/SoftmaxSoftmax'sequential_1/fruit_tn2/Reshape:output:0*
T0*(
_output_shapes
:???????????
1sequential_1/fruit_tn2/ActivityRegularizer/SquareSquare(sequential_1/fruit_tn2/Softmax:softmax:0*
T0*(
_output_shapes
:???????????
0sequential_1/fruit_tn2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
.sequential_1/fruit_tn2/ActivityRegularizer/SumSum5sequential_1/fruit_tn2/ActivityRegularizer/Square:y:09sequential_1/fruit_tn2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: u
0sequential_1/fruit_tn2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
.sequential_1/fruit_tn2/ActivityRegularizer/mulMul9sequential_1/fruit_tn2/ActivityRegularizer/mul/x:output:07sequential_1/fruit_tn2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: ?
0sequential_1/fruit_tn2/ActivityRegularizer/ShapeShape(sequential_1/fruit_tn2/Softmax:softmax:0*
T0*
_output_shapes
:?
>sequential_1/fruit_tn2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
@sequential_1/fruit_tn2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@sequential_1/fruit_tn2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_1/fruit_tn2/ActivityRegularizer/strided_sliceStridedSlice9sequential_1/fruit_tn2/ActivityRegularizer/Shape:output:0Gsequential_1/fruit_tn2/ActivityRegularizer/strided_slice/stack:output:0Isequential_1/fruit_tn2/ActivityRegularizer/strided_slice/stack_1:output:0Isequential_1/fruit_tn2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
/sequential_1/fruit_tn2/ActivityRegularizer/CastCastAsequential_1/fruit_tn2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
2sequential_1/fruit_tn2/ActivityRegularizer/truedivRealDiv2sequential_1/fruit_tn2/ActivityRegularizer/mul:z:03sequential_1/fruit_tn2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: x
IdentityIdentity(sequential_1/fruit_tn2/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp+^sequential_1/conv2d/BiasAdd/ReadVariableOp*^sequential_1/conv2d/Conv2D/ReadVariableOp&^sequential_1/conv2dmpo/ReadVariableOp(^sequential_1/conv2dmpo/ReadVariableOp_10^sequential_1/conv2dmpo/Reshape_2/ReadVariableOp(^sequential_1/conv2dmpo_1/ReadVariableOp*^sequential_1/conv2dmpo_1/ReadVariableOp_1*^sequential_1/conv2dmpo_1/ReadVariableOp_2*^sequential_1/conv2dmpo_1/ReadVariableOp_32^sequential_1/conv2dmpo_1/Reshape_2/ReadVariableOp0^sequential_1/fruit_tn1/loop_body/ReadVariableOp2^sequential_1/fruit_tn1/loop_body/ReadVariableOp_12^sequential_1/fruit_tn1/loop_body/ReadVariableOp_24^sequential_1/fruit_tn1/loop_body/add/ReadVariableOp0^sequential_1/fruit_tn2/loop_body/ReadVariableOp2^sequential_1/fruit_tn2/loop_body/ReadVariableOp_12^sequential_1/fruit_tn2/loop_body/ReadVariableOp_24^sequential_1/fruit_tn2/loop_body/add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????22: : : : : : : : : : : : : : : : : : 2X
*sequential_1/conv2d/BiasAdd/ReadVariableOp*sequential_1/conv2d/BiasAdd/ReadVariableOp2V
)sequential_1/conv2d/Conv2D/ReadVariableOp)sequential_1/conv2d/Conv2D/ReadVariableOp2N
%sequential_1/conv2dmpo/ReadVariableOp%sequential_1/conv2dmpo/ReadVariableOp2R
'sequential_1/conv2dmpo/ReadVariableOp_1'sequential_1/conv2dmpo/ReadVariableOp_12b
/sequential_1/conv2dmpo/Reshape_2/ReadVariableOp/sequential_1/conv2dmpo/Reshape_2/ReadVariableOp2R
'sequential_1/conv2dmpo_1/ReadVariableOp'sequential_1/conv2dmpo_1/ReadVariableOp2V
)sequential_1/conv2dmpo_1/ReadVariableOp_1)sequential_1/conv2dmpo_1/ReadVariableOp_12V
)sequential_1/conv2dmpo_1/ReadVariableOp_2)sequential_1/conv2dmpo_1/ReadVariableOp_22V
)sequential_1/conv2dmpo_1/ReadVariableOp_3)sequential_1/conv2dmpo_1/ReadVariableOp_32f
1sequential_1/conv2dmpo_1/Reshape_2/ReadVariableOp1sequential_1/conv2dmpo_1/Reshape_2/ReadVariableOp2b
/sequential_1/fruit_tn1/loop_body/ReadVariableOp/sequential_1/fruit_tn1/loop_body/ReadVariableOp2f
1sequential_1/fruit_tn1/loop_body/ReadVariableOp_11sequential_1/fruit_tn1/loop_body/ReadVariableOp_12f
1sequential_1/fruit_tn1/loop_body/ReadVariableOp_21sequential_1/fruit_tn1/loop_body/ReadVariableOp_22j
3sequential_1/fruit_tn1/loop_body/add/ReadVariableOp3sequential_1/fruit_tn1/loop_body/add/ReadVariableOp2b
/sequential_1/fruit_tn2/loop_body/ReadVariableOp/sequential_1/fruit_tn2/loop_body/ReadVariableOp2f
1sequential_1/fruit_tn2/loop_body/ReadVariableOp_11sequential_1/fruit_tn2/loop_body/ReadVariableOp_12f
1sequential_1/fruit_tn2/loop_body/ReadVariableOp_21sequential_1/fruit_tn2/loop_body/ReadVariableOp_22j
3sequential_1/fruit_tn2/loop_body/add/ReadVariableOp3sequential_1/fruit_tn2/loop_body/add/ReadVariableOp:Y U
/
_output_shapes
:?????????22
"
_user_specified_name
input_12
??
?
E__inference_fruit_tn2_layer_call_and_return_conditional_losses_413174

inputs3
!loop_body_readvariableop_resource:>
#loop_body_readvariableop_1_resource:?5
#loop_body_readvariableop_2_resource:4
%loop_body_add_readvariableop_resource:	?
identity??loop_body/ReadVariableOp?loop_body/ReadVariableOp_1?loop_body/ReadVariableOp_2?loop_body/add/ReadVariableOp;
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
Tparams0*
_output_shapes
:@l
loop_body/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
loop_body/ReshapeReshapeloop_body/GatherV2:output:0 loop_body/Reshape/shape:output:0*
T0*"
_output_shapes
:z
loop_body/ReadVariableOpReadVariableOp!loop_body_readvariableop_resource*
_output_shapes

:*
dtype0?
loop_body/ReadVariableOp_1ReadVariableOp#loop_body_readvariableop_1_resource*'
_output_shapes
:?*
dtype0~
loop_body/ReadVariableOp_2ReadVariableOp#loop_body_readvariableop_2_resource*
_output_shapes

:*
dtype0w
"loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
loop_body/Tensordot/transpose	Transposeloop_body/Reshape:output:0+loop_body/Tensordot/transpose/perm:output:0*
T0*"
_output_shapes
:r
!loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
loop_body/Tensordot/ReshapeReshape!loop_body/Tensordot/transpose:y:0*loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:?
loop_body/Tensordot/MatMulMatMul$loop_body/Tensordot/Reshape:output:0 loop_body/ReadVariableOp:value:0*
T0*
_output_shapes

:n
loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
loop_body/TensordotReshape$loop_body/Tensordot/MatMul:product:0"loop_body/Tensordot/shape:output:0*
T0*"
_output_shapes
:}
$loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
loop_body/Tensordot_1/transpose	Transpose"loop_body/ReadVariableOp_1:value:0-loop_body/Tensordot_1/transpose/perm:output:0*
T0*'
_output_shapes
:?t
#loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"     ?
loop_body/Tensordot_1/ReshapeReshape#loop_body/Tensordot_1/transpose:y:0,loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes
:	?{
&loop_body/Tensordot_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
!loop_body/Tensordot_1/transpose_1	Transposeloop_body/Tensordot:output:0/loop_body/Tensordot_1/transpose_1/perm:output:0*
T0*"
_output_shapes
:v
%loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
loop_body/Tensordot_1/Reshape_1Reshape%loop_body/Tensordot_1/transpose_1:y:0.loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:?
loop_body/Tensordot_1/MatMulMatMul&loop_body/Tensordot_1/Reshape:output:0(loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	?p
loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"?         ?
loop_body/Tensordot_1Reshape&loop_body/Tensordot_1/MatMul:product:0$loop_body/Tensordot_1/shape:output:0*
T0*#
_output_shapes
:?t
#loop_body/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
loop_body/Tensordot_2/ReshapeReshape"loop_body/ReadVariableOp_2:value:0,loop_body/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes

:y
$loop_body/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
loop_body/Tensordot_2/transpose	Transposeloop_body/Tensordot_1:output:0-loop_body/Tensordot_2/transpose/perm:output:0*
T0*#
_output_shapes
:?v
%loop_body/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   ?
loop_body/Tensordot_2/Reshape_1Reshape#loop_body/Tensordot_2/transpose:y:0.loop_body/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	??
loop_body/Tensordot_2/MatMulMatMul&loop_body/Tensordot_2/Reshape:output:0(loop_body/Tensordot_2/Reshape_1:output:0*
T0*
_output_shapes
:	?f
loop_body/Tensordot_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:??
loop_body/Tensordot_2Reshape&loop_body/Tensordot_2/MatMul:product:0$loop_body/Tensordot_2/shape:output:0*
T0*
_output_shapes	
:?
loop_body/add/ReadVariableOpReadVariableOp%loop_body_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
loop_body/addAddV2loop_body/Tensordot_2:output:0$loop_body/add/ReadVariableOp:value:0*
T0*
_output_shapes	
:?\
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
Tparams0*'
_output_shapes
:?????????@d
"loop_body/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
loop_body/Reshape/pfor/concatConcatV2pfor/Reshape:output:0 loop_body/Reshape/shape:output:0+loop_body/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
loop_body/Reshape/pfor/ReshapeReshape)loop_body/GatherV2/pfor/GatherV2:output:0&loop_body/Reshape/pfor/concat:output:0*
T0*/
_output_shapes
:?????????j
(loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
&loop_body/Tensordot/transpose/pfor/addAddV2+loop_body/Tensordot/transpose/perm:output:01loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:|
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
:?
,loop_body/Tensordot/transpose/pfor/Transpose	Transpose'loop_body/Reshape/pfor/Reshape:output:02loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:?????????n
,loop_body/Tensordot/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'loop_body/Tensordot/Reshape/pfor/concatConcatV2pfor/Reshape:output:0*loop_body/Tensordot/Reshape/shape:output:05loop_body/Tensordot/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
(loop_body/Tensordot/Reshape/pfor/ReshapeReshape0loop_body/Tensordot/transpose/pfor/Transpose:y:00loop_body/Tensordot/Reshape/pfor/concat:output:0*
T0*+
_output_shapes
:??????????
%loop_body/Tensordot/MatMul/pfor/ShapeShape1loop_body/Tensordot/Reshape/pfor/Reshape:output:0*
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
)loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape1loop_body/Tensordot/Reshape/pfor/Reshape:output:08loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:???????????????????
&loop_body/Tensordot/MatMul/pfor/MatMulMatMul2loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0 loop_body/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
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
T0*+
_output_shapes
:?????????f
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
:?????????n
,loop_body/Tensordot_1/transpose_1/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
*loop_body/Tensordot_1/transpose_1/pfor/addAddV2/loop_body/Tensordot_1/transpose_1/perm:output:05loop_body/Tensordot_1/transpose_1/pfor/add/y:output:0*
T0*
_output_shapes
:?
6loop_body/Tensordot_1/transpose_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: t
2loop_body/Tensordot_1/transpose_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-loop_body/Tensordot_1/transpose_1/pfor/concatConcatV2?loop_body/Tensordot_1/transpose_1/pfor/concat/values_0:output:0.loop_body/Tensordot_1/transpose_1/pfor/add:z:0;loop_body/Tensordot_1/transpose_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
0loop_body/Tensordot_1/transpose_1/pfor/Transpose	Transpose)loop_body/Tensordot/pfor/Reshape:output:06loop_body/Tensordot_1/transpose_1/pfor/concat:output:0*
T0*/
_output_shapes
:?????????r
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
,loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape4loop_body/Tensordot_1/transpose_1/pfor/Transpose:y:04loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*+
_output_shapes
:??????????
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
T0*+
_output_shapes
:??????????
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
T0*(
_output_shapes
:??????????~
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
T0*,
_output_shapes
:??????????h
&loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
!loop_body/Tensordot_1/pfor/concatConcatV2pfor/Reshape:output:0$loop_body/Tensordot_1/shape:output:0/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
"loop_body/Tensordot_1/pfor/ReshapeReshape1loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:0*loop_body/Tensordot_1/pfor/concat:output:0*
T0*0
_output_shapes
:??????????l
*loop_body/Tensordot_2/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
(loop_body/Tensordot_2/transpose/pfor/addAddV2-loop_body/Tensordot_2/transpose/perm:output:03loop_body/Tensordot_2/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:~
4loop_body/Tensordot_2/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: r
0loop_body/Tensordot_2/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+loop_body/Tensordot_2/transpose/pfor/concatConcatV2=loop_body/Tensordot_2/transpose/pfor/concat/values_0:output:0,loop_body/Tensordot_2/transpose/pfor/add:z:09loop_body/Tensordot_2/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
.loop_body/Tensordot_2/transpose/pfor/Transpose	Transpose+loop_body/Tensordot_1/pfor/Reshape:output:04loop_body/Tensordot_2/transpose/pfor/concat:output:0*
T0*0
_output_shapes
:??????????r
0loop_body/Tensordot_2/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+loop_body/Tensordot_2/Reshape_1/pfor/concatConcatV2pfor/Reshape:output:0.loop_body/Tensordot_2/Reshape_1/shape:output:09loop_body/Tensordot_2/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,loop_body/Tensordot_2/Reshape_1/pfor/ReshapeReshape2loop_body/Tensordot_2/transpose/pfor/Transpose:y:04loop_body/Tensordot_2/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:???????????
'loop_body/Tensordot_2/MatMul/pfor/ShapeShape5loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
5loop_body/Tensordot_2/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/loop_body/Tensordot_2/MatMul/pfor/strided_sliceStridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0>loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_1StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
)loop_body/Tensordot_2/MatMul/pfor/MinimumMinimum8loop_body/Tensordot_2/MatMul/pfor/strided_slice:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: k
)loop_body/Tensordot_2/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :?
'loop_body/Tensordot_2/MatMul/pfor/EqualEqual-loop_body/Tensordot_2/MatMul/pfor/Minimum:z:02loop_body/Tensordot_2/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: 
*loop_body/Tensordot_2/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          
*loop_body/Tensordot_2/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          ?
(loop_body/Tensordot_2/MatMul/pfor/SelectSelect+loop_body/Tensordot_2/MatMul/pfor/Equal:z:03loop_body/Tensordot_2/MatMul/pfor/Select/t:output:03loop_body/Tensordot_2/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_2StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_3StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_4StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
'loop_body/Tensordot_2/MatMul/pfor/stackPack:loop_body/Tensordot_2/MatMul/pfor/strided_slice_2:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_3:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
+loop_body/Tensordot_2/MatMul/pfor/transpose	Transpose5loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:01loop_body/Tensordot_2/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'????????????????????????????
)loop_body/Tensordot_2/MatMul/pfor/ReshapeReshape/loop_body/Tensordot_2/MatMul/pfor/transpose:y:00loop_body/Tensordot_2/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:???????????
)loop_body/Tensordot_2/MatMul/pfor/Shape_1Shape2loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:s
1loop_body/Tensordot_2/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
'loop_body/Tensordot_2/MatMul/pfor/splitSplit:loop_body/Tensordot_2/MatMul/pfor/split/split_dim:output:02loop_body/Tensordot_2/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_splitt
1loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB v
3loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
+loop_body/Tensordot_2/MatMul/pfor/Reshape_1Reshape0loop_body/Tensordot_2/MatMul/pfor/split:output:0<loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: t
1loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB v
3loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
+loop_body/Tensordot_2/MatMul/pfor/Reshape_2Reshape0loop_body/Tensordot_2/MatMul/pfor/split:output:1<loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: t
1loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB v
3loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
+loop_body/Tensordot_2/MatMul/pfor/Reshape_3Reshape0loop_body/Tensordot_2/MatMul/pfor/split:output:2<loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ?
%loop_body/Tensordot_2/MatMul/pfor/mulMul4loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ?
1loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shapePack4loop_body/Tensordot_2/MatMul/pfor/Reshape_1:output:0)loop_body/Tensordot_2/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:?
+loop_body/Tensordot_2/MatMul/pfor/Reshape_4Reshape2loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0:loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:???????????????????
(loop_body/Tensordot_2/MatMul/pfor/MatMulMatMul&loop_body/Tensordot_2/Reshape:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:?????????~
3loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
??????????
1loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shapePack<loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:?
+loop_body/Tensordot_2/MatMul/pfor/Reshape_5Reshape2loop_body/Tensordot_2/MatMul/pfor/MatMul:product:0:loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'????????????????????????????
2loop_body/Tensordot_2/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
-loop_body/Tensordot_2/MatMul/pfor/transpose_1	Transpose4loop_body/Tensordot_2/MatMul/pfor/Reshape_5:output:0;loop_body/Tensordot_2/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????h
&loop_body/Tensordot_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
!loop_body/Tensordot_2/pfor/concatConcatV2pfor/Reshape:output:0$loop_body/Tensordot_2/shape:output:0/loop_body/Tensordot_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
"loop_body/Tensordot_2/pfor/ReshapeReshape1loop_body/Tensordot_2/MatMul/pfor/transpose_1:y:0*loop_body/Tensordot_2/pfor/concat:output:0*
T0*(
_output_shapes
:??????????Y
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
value	B :?
loop_body/add/pfor/addAddV2"loop_body/add/pfor/Rank_1:output:0!loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: ?
loop_body/add/pfor/MaximumMaximumloop_body/add/pfor/add:z:0 loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: s
loop_body/add/pfor/ShapeShape+loop_body/Tensordot_2/pfor/Reshape:output:0*
T0*
_output_shapes
:?
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
:*
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
:?
loop_body/add/pfor/Reshape_1Reshape+loop_body/Tensordot_2/pfor/Reshape:output:0"loop_body/add/pfor/concat:output:0*
T0*(
_output_shapes
:???????????
loop_body/add/pfor/AddV2AddV2%loop_body/add/pfor/Reshape_1:output:0$loop_body/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   {
ReshapeReshapeloop_body/add/pfor/AddV2:z:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????W
SoftmaxSoftmaxReshape:output:0*
T0*(
_output_shapes
:??????????a
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^loop_body/ReadVariableOp^loop_body/ReadVariableOp_1^loop_body/ReadVariableOp_2^loop_body/add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@: : : : 24
loop_body/ReadVariableOploop_body/ReadVariableOp28
loop_body/ReadVariableOp_1loop_body/ReadVariableOp_128
loop_body/ReadVariableOp_2loop_body/ReadVariableOp_22<
loop_body/add/ReadVariableOploop_body/add/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
H
1__inference_fruit_tn2_activity_regularizer_412353
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
?_
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_413780
input_12'
conv2d_413699:
conv2d_413701:*
conv2dmpo_413704:*
conv2dmpo_413706:
conv2dmpo_413708:,
conv2dmpo_1_413720:,
conv2dmpo_1_413722:,
conv2dmpo_1_413724:,
conv2dmpo_1_413726:!
conv2dmpo_1_413728:	?&
fruit_tn1_413740:+
fruit_tn1_413742:?&
fruit_tn1_413744:&
fruit_tn1_413746:"
fruit_tn2_413758:+
fruit_tn2_413760:?"
fruit_tn2_413762:
fruit_tn2_413764:	?
identity

identity_1

identity_2

identity_3

identity_4??conv2d/StatefulPartitionedCall?!conv2dmpo/StatefulPartitionedCall?#conv2dmpo_1/StatefulPartitionedCall? dropout1/StatefulPartitionedCall?!fruit_tn1/StatefulPartitionedCall?!fruit_tn2/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_12conv2d_413699conv2d_413701*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_412370?
!conv2dmpo/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2dmpo_413704conv2dmpo_413706conv2dmpo_413708*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2dmpo_layer_call_and_return_conditional_losses_412438?
-conv2dmpo/ActivityRegularizer/PartitionedCallPartitionedCall*conv2dmpo/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *:
f5R3
1__inference_conv2dmpo_activity_regularizer_412290}
#conv2dmpo/ActivityRegularizer/ShapeShape*conv2dmpo/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1conv2dmpo/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3conv2dmpo/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3conv2dmpo/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+conv2dmpo/ActivityRegularizer/strided_sliceStridedSlice,conv2dmpo/ActivityRegularizer/Shape:output:0:conv2dmpo/ActivityRegularizer/strided_slice/stack:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_1:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"conv2dmpo/ActivityRegularizer/CastCast4conv2dmpo/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%conv2dmpo/ActivityRegularizer/truedivRealDiv6conv2dmpo/ActivityRegularizer/PartitionedCall:output:0&conv2dmpo/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
max_pooling2d/PartitionedCallPartitionedCall*conv2dmpo/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_412299?
#conv2dmpo_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2dmpo_1_413720conv2dmpo_1_413722conv2dmpo_1_413724conv2dmpo_1_413726conv2dmpo_1_413728*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2dmpo_1_layer_call_and_return_conditional_losses_412543?
/conv2dmpo_1/ActivityRegularizer/PartitionedCallPartitionedCall,conv2dmpo_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *<
f7R5
3__inference_conv2dmpo_1_activity_regularizer_412315?
%conv2dmpo_1/ActivityRegularizer/ShapeShape,conv2dmpo_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:}
3conv2dmpo_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5conv2dmpo_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5conv2dmpo_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-conv2dmpo_1/ActivityRegularizer/strided_sliceStridedSlice.conv2dmpo_1/ActivityRegularizer/Shape:output:0<conv2dmpo_1/ActivityRegularizer/strided_slice/stack:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_1:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
$conv2dmpo_1/ActivityRegularizer/CastCast6conv2dmpo_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
'conv2dmpo_1/ActivityRegularizer/truedivRealDiv8conv2dmpo_1/ActivityRegularizer/PartitionedCall:output:0(conv2dmpo_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
max_pooling2d_1/PartitionedCallPartitionedCall,conv2dmpo_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_412324?
!fruit_tn1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0fruit_tn1_413740fruit_tn1_413742fruit_tn1_413744fruit_tn1_413746*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_fruit_tn1_layer_call_and_return_conditional_losses_412856?
-fruit_tn1/ActivityRegularizer/PartitionedCallPartitionedCall*fruit_tn1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *:
f5R3
1__inference_fruit_tn1_activity_regularizer_412340}
#fruit_tn1/ActivityRegularizer/ShapeShape*fruit_tn1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1fruit_tn1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3fruit_tn1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3fruit_tn1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+fruit_tn1/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn1/ActivityRegularizer/Shape:output:0:fruit_tn1/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"fruit_tn1/ActivityRegularizer/CastCast4fruit_tn1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%fruit_tn1/ActivityRegularizer/truedivRealDiv6fruit_tn1/ActivityRegularizer/PartitionedCall:output:0&fruit_tn1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
 dropout1/StatefulPartitionedCallStatefulPartitionedCall*fruit_tn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout1_layer_call_and_return_conditional_losses_413290?
!fruit_tn2/StatefulPartitionedCallStatefulPartitionedCall)dropout1/StatefulPartitionedCall:output:0fruit_tn2_413758fruit_tn2_413760fruit_tn2_413762fruit_tn2_413764*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_fruit_tn2_layer_call_and_return_conditional_losses_413174?
-fruit_tn2/ActivityRegularizer/PartitionedCallPartitionedCall*fruit_tn2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *:
f5R3
1__inference_fruit_tn2_activity_regularizer_412353}
#fruit_tn2/ActivityRegularizer/ShapeShape*fruit_tn2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1fruit_tn2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3fruit_tn2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3fruit_tn2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+fruit_tn2/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn2/ActivityRegularizer/Shape:output:0:fruit_tn2/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"fruit_tn2/ActivityRegularizer/CastCast4fruit_tn2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%fruit_tn2/ActivityRegularizer/truedivRealDiv6fruit_tn2/ActivityRegularizer/PartitionedCall:output:0&fruit_tn2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: z
IdentityIdentity*fruit_tn2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????i

Identity_1Identity)conv2dmpo/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: k

Identity_2Identity+conv2dmpo_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_3Identity)fruit_tn1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_4Identity)fruit_tn2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^conv2d/StatefulPartitionedCall"^conv2dmpo/StatefulPartitionedCall$^conv2dmpo_1/StatefulPartitionedCall!^dropout1/StatefulPartitionedCall"^fruit_tn1/StatefulPartitionedCall"^fruit_tn2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????22: : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2F
!conv2dmpo/StatefulPartitionedCall!conv2dmpo/StatefulPartitionedCall2J
#conv2dmpo_1/StatefulPartitionedCall#conv2dmpo_1/StatefulPartitionedCall2D
 dropout1/StatefulPartitionedCall dropout1/StatefulPartitionedCall2F
!fruit_tn1/StatefulPartitionedCall!fruit_tn1/StatefulPartitionedCall2F
!fruit_tn2/StatefulPartitionedCall!fruit_tn2/StatefulPartitionedCall:Y U
/
_output_shapes
:?????????22
"
_user_specified_name
input_12
?
?
'__inference_conv2d_layer_call_fn_415549

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_412370w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????//`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????22: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????22
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_415540
input_12!
unknown:
	unknown_0:#
	unknown_1:#
	unknown_2:
	unknown_3:#
	unknown_4:#
	unknown_5:#
	unknown_6:#
	unknown_7:
	unknown_8:	?
	unknown_9:%

unknown_10:? 

unknown_11: 

unknown_12:

unknown_13:%

unknown_14:?

unknown_15:

unknown_16:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_12unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_412277p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????22: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????22
"
_user_specified_name
input_12
??
? 
__inference__traced_save_416732
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop7
3savev2_conv2dmpo_end_node_first_read_readvariableop6
2savev2_conv2dmpo_end_node_last_read_readvariableop-
)savev2_conv2dmpo_bias_read_readvariableop9
5savev2_conv2dmpo_1_end_node_first_read_readvariableop8
4savev2_conv2dmpo_1_middle_node_0_read_readvariableop8
4savev2_conv2dmpo_1_middle_node_1_read_readvariableop8
4savev2_conv2dmpo_1_end_node_last_read_readvariableop/
+savev2_conv2dmpo_1_bias_read_readvariableop 
savev2_a_read_readvariableop 
savev2_b_read_readvariableop 
savev2_c_read_readvariableop#
savev2_bias_read_readvariableop"
savev2_a_1_read_readvariableop"
savev2_b_1_read_readvariableop"
savev2_c_1_read_readvariableop%
!savev2_bias_1_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop>
:savev2_adam_conv2dmpo_end_node_first_m_read_readvariableop=
9savev2_adam_conv2dmpo_end_node_last_m_read_readvariableop4
0savev2_adam_conv2dmpo_bias_m_read_readvariableop@
<savev2_adam_conv2dmpo_1_end_node_first_m_read_readvariableop?
;savev2_adam_conv2dmpo_1_middle_node_0_m_read_readvariableop?
;savev2_adam_conv2dmpo_1_middle_node_1_m_read_readvariableop?
;savev2_adam_conv2dmpo_1_end_node_last_m_read_readvariableop6
2savev2_adam_conv2dmpo_1_bias_m_read_readvariableop'
#savev2_adam_a_m_read_readvariableop'
#savev2_adam_b_m_read_readvariableop'
#savev2_adam_c_m_read_readvariableop*
&savev2_adam_bias_m_read_readvariableop)
%savev2_adam_a_m_1_read_readvariableop)
%savev2_adam_b_m_1_read_readvariableop)
%savev2_adam_c_m_1_read_readvariableop,
(savev2_adam_bias_m_1_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop>
:savev2_adam_conv2dmpo_end_node_first_v_read_readvariableop=
9savev2_adam_conv2dmpo_end_node_last_v_read_readvariableop4
0savev2_adam_conv2dmpo_bias_v_read_readvariableop@
<savev2_adam_conv2dmpo_1_end_node_first_v_read_readvariableop?
;savev2_adam_conv2dmpo_1_middle_node_0_v_read_readvariableop?
;savev2_adam_conv2dmpo_1_middle_node_1_v_read_readvariableop?
;savev2_adam_conv2dmpo_1_end_node_last_v_read_readvariableop6
2savev2_adam_conv2dmpo_1_bias_v_read_readvariableop'
#savev2_adam_a_v_read_readvariableop'
#savev2_adam_b_v_read_readvariableop'
#savev2_adam_c_v_read_readvariableop*
&savev2_adam_bias_v_read_readvariableop)
%savev2_adam_a_v_1_read_readvariableop)
%savev2_adam_b_v_1_read_readvariableop)
%savev2_adam_c_v_1_read_readvariableop,
(savev2_adam_bias_v_1_read_readvariableop6
2savev2_adam_conv2d_kernel_vhat_read_readvariableop4
0savev2_adam_conv2d_bias_vhat_read_readvariableopA
=savev2_adam_conv2dmpo_end_node_first_vhat_read_readvariableop@
<savev2_adam_conv2dmpo_end_node_last_vhat_read_readvariableop7
3savev2_adam_conv2dmpo_bias_vhat_read_readvariableopC
?savev2_adam_conv2dmpo_1_end_node_first_vhat_read_readvariableopB
>savev2_adam_conv2dmpo_1_middle_node_0_vhat_read_readvariableopB
>savev2_adam_conv2dmpo_1_middle_node_1_vhat_read_readvariableopB
>savev2_adam_conv2dmpo_1_end_node_last_vhat_read_readvariableop9
5savev2_adam_conv2dmpo_1_bias_vhat_read_readvariableop*
&savev2_adam_a_vhat_read_readvariableop*
&savev2_adam_b_vhat_read_readvariableop*
&savev2_adam_c_vhat_read_readvariableop-
)savev2_adam_bias_vhat_read_readvariableop,
(savev2_adam_a_vhat_1_read_readvariableop,
(savev2_adam_b_vhat_1_read_readvariableop,
(savev2_adam_c_vhat_1_read_readvariableop/
+savev2_adam_bias_vhat_1_read_readvariableop
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
: ?0
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*?0
value?0B?0QB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/end_node_first/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-1/end_node_last/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/end_node_first/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-2/middle_node_0/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-2/middle_node_1/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-2/end_node_last/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/a_var/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/b_var/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/c_var/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/a_var/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/b_var/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/c_var/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/end_node_first/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/end_node_last/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/end_node_first/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/middle_node_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/middle_node_1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/end_node_last/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/a_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/b_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/c_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/a_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/b_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/c_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/end_node_first/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/end_node_last/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/end_node_first/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/middle_node_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/middle_node_1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/end_node_last/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/a_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/b_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/c_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/a_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/b_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/c_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-1/end_node_first/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/end_node_last/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-2/end_node_first/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/middle_node_0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/middle_node_1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/end_node_last/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/a_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/b_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/c_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/a_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/b_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/c_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*?
value?B?QB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop3savev2_conv2dmpo_end_node_first_read_readvariableop2savev2_conv2dmpo_end_node_last_read_readvariableop)savev2_conv2dmpo_bias_read_readvariableop5savev2_conv2dmpo_1_end_node_first_read_readvariableop4savev2_conv2dmpo_1_middle_node_0_read_readvariableop4savev2_conv2dmpo_1_middle_node_1_read_readvariableop4savev2_conv2dmpo_1_end_node_last_read_readvariableop+savev2_conv2dmpo_1_bias_read_readvariableopsavev2_a_read_readvariableopsavev2_b_read_readvariableopsavev2_c_read_readvariableopsavev2_bias_read_readvariableopsavev2_a_1_read_readvariableopsavev2_b_1_read_readvariableopsavev2_c_1_read_readvariableop!savev2_bias_1_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop:savev2_adam_conv2dmpo_end_node_first_m_read_readvariableop9savev2_adam_conv2dmpo_end_node_last_m_read_readvariableop0savev2_adam_conv2dmpo_bias_m_read_readvariableop<savev2_adam_conv2dmpo_1_end_node_first_m_read_readvariableop;savev2_adam_conv2dmpo_1_middle_node_0_m_read_readvariableop;savev2_adam_conv2dmpo_1_middle_node_1_m_read_readvariableop;savev2_adam_conv2dmpo_1_end_node_last_m_read_readvariableop2savev2_adam_conv2dmpo_1_bias_m_read_readvariableop#savev2_adam_a_m_read_readvariableop#savev2_adam_b_m_read_readvariableop#savev2_adam_c_m_read_readvariableop&savev2_adam_bias_m_read_readvariableop%savev2_adam_a_m_1_read_readvariableop%savev2_adam_b_m_1_read_readvariableop%savev2_adam_c_m_1_read_readvariableop(savev2_adam_bias_m_1_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop:savev2_adam_conv2dmpo_end_node_first_v_read_readvariableop9savev2_adam_conv2dmpo_end_node_last_v_read_readvariableop0savev2_adam_conv2dmpo_bias_v_read_readvariableop<savev2_adam_conv2dmpo_1_end_node_first_v_read_readvariableop;savev2_adam_conv2dmpo_1_middle_node_0_v_read_readvariableop;savev2_adam_conv2dmpo_1_middle_node_1_v_read_readvariableop;savev2_adam_conv2dmpo_1_end_node_last_v_read_readvariableop2savev2_adam_conv2dmpo_1_bias_v_read_readvariableop#savev2_adam_a_v_read_readvariableop#savev2_adam_b_v_read_readvariableop#savev2_adam_c_v_read_readvariableop&savev2_adam_bias_v_read_readvariableop%savev2_adam_a_v_1_read_readvariableop%savev2_adam_b_v_1_read_readvariableop%savev2_adam_c_v_1_read_readvariableop(savev2_adam_bias_v_1_read_readvariableop2savev2_adam_conv2d_kernel_vhat_read_readvariableop0savev2_adam_conv2d_bias_vhat_read_readvariableop=savev2_adam_conv2dmpo_end_node_first_vhat_read_readvariableop<savev2_adam_conv2dmpo_end_node_last_vhat_read_readvariableop3savev2_adam_conv2dmpo_bias_vhat_read_readvariableop?savev2_adam_conv2dmpo_1_end_node_first_vhat_read_readvariableop>savev2_adam_conv2dmpo_1_middle_node_0_vhat_read_readvariableop>savev2_adam_conv2dmpo_1_middle_node_1_vhat_read_readvariableop>savev2_adam_conv2dmpo_1_end_node_last_vhat_read_readvariableop5savev2_adam_conv2dmpo_1_bias_vhat_read_readvariableop&savev2_adam_a_vhat_read_readvariableop&savev2_adam_b_vhat_read_readvariableop&savev2_adam_c_vhat_read_readvariableop)savev2_adam_bias_vhat_read_readvariableop(savev2_adam_a_vhat_1_read_readvariableop(savev2_adam_b_vhat_1_read_readvariableop(savev2_adam_c_vhat_1_read_readvariableop+savev2_adam_bias_vhat_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *_
dtypesU
S2Q	?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::::::?::?::::?::?: : : : : : ::::::::::::?::?::::?::?::::::::::?::?::::?::?::::::::::?::?::::?::?: 2(
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
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::,(
&
_output_shapes
::,(
&
_output_shapes
::,	(
&
_output_shapes
::!


_output_shapes	
:?:($
"
_output_shapes
::-)
'
_output_shapes
:?:($
"
_output_shapes
::($
"
_output_shapes
::$ 

_output_shapes

::-)
'
_output_shapes
:?:$ 

_output_shapes

::!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
::,!(
&
_output_shapes
::,"(
&
_output_shapes
::,#(
&
_output_shapes
::!$

_output_shapes	
:?:(%$
"
_output_shapes
::-&)
'
_output_shapes
:?:('$
"
_output_shapes
::(($
"
_output_shapes
::$) 

_output_shapes

::-*)
'
_output_shapes
:?:$+ 

_output_shapes

::!,

_output_shapes	
:?:,-(
&
_output_shapes
:: .

_output_shapes
::,/(
&
_output_shapes
::,0(
&
_output_shapes
:: 1

_output_shapes
::,2(
&
_output_shapes
::,3(
&
_output_shapes
::,4(
&
_output_shapes
::,5(
&
_output_shapes
::!6

_output_shapes	
:?:(7$
"
_output_shapes
::-8)
'
_output_shapes
:?:(9$
"
_output_shapes
::(:$
"
_output_shapes
::$; 

_output_shapes

::-<)
'
_output_shapes
:?:$= 

_output_shapes

::!>

_output_shapes	
:?:,?(
&
_output_shapes
:: @

_output_shapes
::,A(
&
_output_shapes
::,B(
&
_output_shapes
:: C

_output_shapes
::,D(
&
_output_shapes
::,E(
&
_output_shapes
::,F(
&
_output_shapes
::,G(
&
_output_shapes
::!H

_output_shapes	
:?:(I$
"
_output_shapes
::-J)
'
_output_shapes
:?:(K$
"
_output_shapes
::(L$
"
_output_shapes
::$M 

_output_shapes

::-N)
'
_output_shapes
:?:$O 

_output_shapes

::!P

_output_shapes	
:?:Q

_output_shapes
: 
?
?
-__inference_sequential_1_layer_call_fn_413906

inputs!
unknown:
	unknown_0:#
	unknown_1:#
	unknown_2:
	unknown_3:#
	unknown_4:#
	unknown_5:#
	unknown_6:#
	unknown_7:
	unknown_8:	?
	unknown_9:%

unknown_10:? 

unknown_11: 

unknown_12:

unknown_13:%

unknown_14:?

unknown_15:

unknown_16:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout	
2*
_collective_manager_ids
 *0
_output_shapes
:??????????: : : : *4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_413524p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????22: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????22
 
_user_specified_nameinputs
?^
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_413197

inputs'
conv2d_412371:
conv2d_412373:*
conv2dmpo_412439:*
conv2dmpo_412441:
conv2dmpo_412443:,
conv2dmpo_1_412544:,
conv2dmpo_1_412546:,
conv2dmpo_1_412548:,
conv2dmpo_1_412550:!
conv2dmpo_1_412552:	?&
fruit_tn1_412857:+
fruit_tn1_412859:?&
fruit_tn1_412861:&
fruit_tn1_412863:"
fruit_tn2_413175:+
fruit_tn2_413177:?"
fruit_tn2_413179:
fruit_tn2_413181:	?
identity

identity_1

identity_2

identity_3

identity_4??conv2d/StatefulPartitionedCall?!conv2dmpo/StatefulPartitionedCall?#conv2dmpo_1/StatefulPartitionedCall?!fruit_tn1/StatefulPartitionedCall?!fruit_tn2/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_412371conv2d_412373*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_412370?
!conv2dmpo/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2dmpo_412439conv2dmpo_412441conv2dmpo_412443*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2dmpo_layer_call_and_return_conditional_losses_412438?
-conv2dmpo/ActivityRegularizer/PartitionedCallPartitionedCall*conv2dmpo/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *:
f5R3
1__inference_conv2dmpo_activity_regularizer_412290}
#conv2dmpo/ActivityRegularizer/ShapeShape*conv2dmpo/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1conv2dmpo/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3conv2dmpo/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3conv2dmpo/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+conv2dmpo/ActivityRegularizer/strided_sliceStridedSlice,conv2dmpo/ActivityRegularizer/Shape:output:0:conv2dmpo/ActivityRegularizer/strided_slice/stack:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_1:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"conv2dmpo/ActivityRegularizer/CastCast4conv2dmpo/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%conv2dmpo/ActivityRegularizer/truedivRealDiv6conv2dmpo/ActivityRegularizer/PartitionedCall:output:0&conv2dmpo/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
max_pooling2d/PartitionedCallPartitionedCall*conv2dmpo/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_412299?
#conv2dmpo_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2dmpo_1_412544conv2dmpo_1_412546conv2dmpo_1_412548conv2dmpo_1_412550conv2dmpo_1_412552*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2dmpo_1_layer_call_and_return_conditional_losses_412543?
/conv2dmpo_1/ActivityRegularizer/PartitionedCallPartitionedCall,conv2dmpo_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *<
f7R5
3__inference_conv2dmpo_1_activity_regularizer_412315?
%conv2dmpo_1/ActivityRegularizer/ShapeShape,conv2dmpo_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:}
3conv2dmpo_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5conv2dmpo_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5conv2dmpo_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-conv2dmpo_1/ActivityRegularizer/strided_sliceStridedSlice.conv2dmpo_1/ActivityRegularizer/Shape:output:0<conv2dmpo_1/ActivityRegularizer/strided_slice/stack:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_1:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
$conv2dmpo_1/ActivityRegularizer/CastCast6conv2dmpo_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
'conv2dmpo_1/ActivityRegularizer/truedivRealDiv8conv2dmpo_1/ActivityRegularizer/PartitionedCall:output:0(conv2dmpo_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
max_pooling2d_1/PartitionedCallPartitionedCall,conv2dmpo_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_412324?
!fruit_tn1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0fruit_tn1_412857fruit_tn1_412859fruit_tn1_412861fruit_tn1_412863*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_fruit_tn1_layer_call_and_return_conditional_losses_412856?
-fruit_tn1/ActivityRegularizer/PartitionedCallPartitionedCall*fruit_tn1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *:
f5R3
1__inference_fruit_tn1_activity_regularizer_412340}
#fruit_tn1/ActivityRegularizer/ShapeShape*fruit_tn1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1fruit_tn1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3fruit_tn1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3fruit_tn1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+fruit_tn1/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn1/ActivityRegularizer/Shape:output:0:fruit_tn1/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"fruit_tn1/ActivityRegularizer/CastCast4fruit_tn1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%fruit_tn1/ActivityRegularizer/truedivRealDiv6fruit_tn1/ActivityRegularizer/PartitionedCall:output:0&fruit_tn1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
dropout1/PartitionedCallPartitionedCall*fruit_tn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout1_layer_call_and_return_conditional_losses_412879?
!fruit_tn2/StatefulPartitionedCallStatefulPartitionedCall!dropout1/PartitionedCall:output:0fruit_tn2_413175fruit_tn2_413177fruit_tn2_413179fruit_tn2_413181*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_fruit_tn2_layer_call_and_return_conditional_losses_413174?
-fruit_tn2/ActivityRegularizer/PartitionedCallPartitionedCall*fruit_tn2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *:
f5R3
1__inference_fruit_tn2_activity_regularizer_412353}
#fruit_tn2/ActivityRegularizer/ShapeShape*fruit_tn2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1fruit_tn2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3fruit_tn2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3fruit_tn2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+fruit_tn2/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn2/ActivityRegularizer/Shape:output:0:fruit_tn2/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"fruit_tn2/ActivityRegularizer/CastCast4fruit_tn2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%fruit_tn2/ActivityRegularizer/truedivRealDiv6fruit_tn2/ActivityRegularizer/PartitionedCall:output:0&fruit_tn2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: z
IdentityIdentity*fruit_tn2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????i

Identity_1Identity)conv2dmpo/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: k

Identity_2Identity+conv2dmpo_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_3Identity)fruit_tn1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_4Identity)fruit_tn2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^conv2d/StatefulPartitionedCall"^conv2dmpo/StatefulPartitionedCall$^conv2dmpo_1/StatefulPartitionedCall"^fruit_tn1/StatefulPartitionedCall"^fruit_tn2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????22: : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2F
!conv2dmpo/StatefulPartitionedCall!conv2dmpo/StatefulPartitionedCall2J
#conv2dmpo_1/StatefulPartitionedCall#conv2dmpo_1/StatefulPartitionedCall2F
!fruit_tn1/StatefulPartitionedCall!fruit_tn1/StatefulPartitionedCall2F
!fruit_tn2/StatefulPartitionedCall!fruit_tn2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????22
 
_user_specified_nameinputs
?
J
.__inference_max_pooling2d_layer_call_fn_415588

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
GPU2*0J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_412299?
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
?
?
I__inference_fruit_tn2_layer_call_and_return_all_conditional_losses_415734

inputs
unknown:$
	unknown_0:?
	unknown_1:
	unknown_2:	?
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_fruit_tn2_layer_call_and_return_conditional_losses_413174?
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
GPU2*0J 8? *:
f5R3
1__inference_fruit_tn2_activity_regularizer_412353p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????X

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
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
,__inference_conv2dmpo_1_layer_call_fn_415608

inputs!
unknown:#
	unknown_0:#
	unknown_1:#
	unknown_2:
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2dmpo_1_layer_call_and_return_conditional_losses_412543x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?U
?
G__inference_conv2dmpo_1_layer_call_and_return_conditional_losses_415884

inputs1
readvariableop_resource:3
readvariableop_1_resource:3
readvariableop_2_resource:3
readvariableop_3_resource:0
!reshape_2_readvariableop_resource:	?
identity??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?Reshape_2/ReadVariableOpn
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0r
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*&
_output_shapes
:*
dtype0r
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*&
_output_shapes
:*
dtype0r
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*&
_output_shapes
:*
dtype0q
Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Tensordot/transpose	TransposeReadVariableOp_1:value:0!Tensordot/transpose/perm:output:0*
T0*&
_output_shapes
:h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@      ?
Tensordot/ReshapeReshapeTensordot/transpose:y:0 Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@s
Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Tensordot/transpose_1	TransposeReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*&
_output_shapes
:j
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:}
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*
_output_shapes

:@p
Tensordot/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  ?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*.
_output_shapes
:s
Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Tensordot_1/transpose	TransposeReadVariableOp_3:value:0#Tensordot_1/transpose/perm:output:0*
T0*&
_output_shapes
:j
Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
Tensordot_1/ReshapeReshapeTensordot_1/transpose:y:0"Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:u
Tensordot_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Tensordot_1/transpose_1	TransposeReadVariableOp_2:value:0%Tensordot_1/transpose_1/perm:output:0*
T0*&
_output_shapes
:l
Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   ?
Tensordot_1/Reshape_1ReshapeTensordot_1/transpose_1:y:0$Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:@?
Tensordot_1/MatMulMatMulTensordot_1/Reshape:output:0Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes

:@r
Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  ?
Tensordot_1ReshapeTensordot_1/MatMul:product:0Tensordot_1/shape:output:0*
T0*.
_output_shapes
:{
Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
Tensordot_2/transpose	TransposeTensordot:output:0#Tensordot_2/transpose/perm:output:0*
T0*.
_output_shapes
:j
Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?      ?
Tensordot_2/ReshapeReshapeTensordot_2/transpose:y:0"Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes
:	?}
Tensordot_2/transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
Tensordot_2/transpose_1	TransposeTensordot_1:output:0%Tensordot_2/transpose_1/perm:output:0*
T0*.
_output_shapes
:l
Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   ?
Tensordot_2/Reshape_1ReshapeTensordot_2/transpose_1:y:0$Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	??
Tensordot_2/MatMulMatMulTensordot_2/Reshape:output:0Tensordot_2/Reshape_1:output:0*
T0* 
_output_shapes
:
???
Tensordot_2/shapeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                              ?
Tensordot_2ReshapeTensordot_2/MatMul:product:0Tensordot_2/shape:output:0*
T0*>
_output_shapes,
*:(
transpose/permConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                         	      ?
	transpose	TransposeTensordot_2:output:0transpose/perm:output:0*
T0*>
_output_shapes,
*:(?
transpose_1/permConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                	               ?
transpose_1	Transposetranspose:y:0transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(v
ShapeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                              ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskO
ConstConst*
_output_shapes
:*
dtype0*
valueB: U
ProdProdstrided_slice:output:0Const:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskT
concat/values_1PackProd:output:0*
N*
T0*
_output_shapes
:V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2strided_slice_1:output:0concat/values_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:q
ReshapeReshapetranspose_1:y:0concat:output:0*
T0*2
_output_shapes 
:u
transpose_2/permConst*
_output_shapes
:*
dtype0*1
value(B&"                      ?
transpose_2	TransposeReshape:output:0transpose_2/perm:output:0*
T0*2
_output_shapes 
:l
Shape_1Const*
_output_shapes
:*
dtype0*1
value(B&"                     _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskQ
Const_1Const*
_output_shapes
:*
dtype0*
valueB: [
Prod_1Prodstrided_slice_2:output:0Const_1:output:0*
T0*
_output_shapes
: _
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskX
concat_1/values_1PackProd_1:output:0*
N*
T0*
_output_shapes
:X
concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concat_1ConcatV2strided_slice_3:output:0concat_1/values_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:j
	Reshape_1Reshapetranspose_2:y:0concat_1:output:0*
T0*'
_output_shapes
:??
Conv2DConv2DinputsReshape_1:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
w
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes	
:?*
dtype0`
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      z
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0*
_output_shapes
:	?l
addAddV2Conv2D:output:0Reshape_2:output:0*
T0*0
_output_shapes
:??????????P
ReluReluadd:z:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^Reshape_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_324
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
1__inference_conv2dmpo_activity_regularizer_412290
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
??
?
E__inference_fruit_tn1_layer_call_and_return_conditional_losses_412856

inputs7
!loop_body_readvariableop_resource:>
#loop_body_readvariableop_1_resource:?9
#loop_body_readvariableop_2_resource:;
%loop_body_add_readvariableop_resource:
identity??loop_body/ReadVariableOp?loop_body/ReadVariableOp_1?loop_body/ReadVariableOp_2?loop_body/add/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
:g
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
Tparams0*#
_output_shapes
:?~
loop_body/ReadVariableOpReadVariableOp!loop_body_readvariableop_resource*"
_output_shapes
:*
dtype0?
loop_body/ReadVariableOp_1ReadVariableOp#loop_body_readvariableop_1_resource*'
_output_shapes
:?*
dtype0?
loop_body/ReadVariableOp_2ReadVariableOp#loop_body_readvariableop_2_resource*"
_output_shapes
:*
dtype0w
"loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
loop_body/Tensordot/transpose	Transposeloop_body/GatherV2:output:0+loop_body/Tensordot/transpose/perm:output:0*
T0*#
_output_shapes
:?r
!loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
loop_body/Tensordot/ReshapeReshape!loop_body/Tensordot/transpose:y:0*loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	?y
$loop_body/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
loop_body/Tensordot/transpose_1	Transpose loop_body/ReadVariableOp:value:0-loop_body/Tensordot/transpose_1/perm:output:0*
T0*"
_output_shapes
:t
#loop_body/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
loop_body/Tensordot/Reshape_1Reshape#loop_body/Tensordot/transpose_1:y:0,loop_body/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:?
loop_body/Tensordot/MatMulMatMul$loop_body/Tensordot/Reshape:output:0&loop_body/Tensordot/Reshape_1:output:0*
T0*
_output_shapes
:	?r
loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            ?
loop_body/TensordotReshape$loop_body/Tensordot/MatMul:product:0"loop_body/Tensordot/shape:output:0*
T0*'
_output_shapes
:?y
$loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
loop_body/Tensordot_1/transpose	Transpose"loop_body/ReadVariableOp_2:value:0-loop_body/Tensordot_1/transpose/perm:output:0*
T0*"
_output_shapes
:t
#loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
loop_body/Tensordot_1/ReshapeReshape#loop_body/Tensordot_1/transpose:y:0,loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:v
%loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
loop_body/Tensordot_1/Reshape_1Reshapeloop_body/Tensordot:output:0.loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?(?
loop_body/Tensordot_1/MatMulMatMul&loop_body/Tensordot_1/Reshape:output:0(loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	?(x
loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"               ?
loop_body/Tensordot_1Reshape&loop_body/Tensordot_1/MatMul:product:0$loop_body/Tensordot_1/shape:output:0*
T0*+
_output_shapes
:?t
#loop_body/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
loop_body/Tensordot_2/ReshapeReshape"loop_body/ReadVariableOp_1:value:0,loop_body/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes
:	?2?
$loop_body/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
loop_body/Tensordot_2/transpose	Transposeloop_body/Tensordot_1:output:0-loop_body/Tensordot_2/transpose/perm:output:0*
T0*+
_output_shapes
:?v
%loop_body/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
loop_body/Tensordot_2/Reshape_1Reshape#loop_body/Tensordot_2/transpose:y:0.loop_body/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2?
loop_body/Tensordot_2/MatMulMatMul&loop_body/Tensordot_2/Reshape:output:0(loop_body/Tensordot_2/Reshape_1:output:0*
T0*
_output_shapes

:p
loop_body/Tensordot_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
loop_body/Tensordot_2Reshape&loop_body/Tensordot_2/MatMul:product:0$loop_body/Tensordot_2/shape:output:0*
T0*"
_output_shapes
:m
loop_body/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
loop_body/transpose	Transposeloop_body/Tensordot_2:output:0!loop_body/transpose/perm:output:0*
T0*"
_output_shapes
:?
loop_body/add/ReadVariableOpReadVariableOp%loop_body_add_readvariableop_resource*"
_output_shapes
:*
dtype0?
loop_body/addAddV2loop_body/transpose:y:0$loop_body/add/ReadVariableOp:value:0*
T0*"
_output_shapes
:\
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
Tparams0*0
_output_shapes
:??????????j
(loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
&loop_body/Tensordot/transpose/pfor/addAddV2+loop_body/Tensordot/transpose/perm:output:01loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:|
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
:?
,loop_body/Tensordot/transpose/pfor/Transpose	Transpose)loop_body/GatherV2/pfor/GatherV2:output:02loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*0
_output_shapes
:??????????n
,loop_body/Tensordot/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'loop_body/Tensordot/Reshape/pfor/concatConcatV2pfor/Reshape:output:0*loop_body/Tensordot/Reshape/shape:output:05loop_body/Tensordot/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
(loop_body/Tensordot/Reshape/pfor/ReshapeReshape0loop_body/Tensordot/transpose/pfor/Transpose:y:00loop_body/Tensordot/Reshape/pfor/concat:output:0*
T0*,
_output_shapes
:???????????
%loop_body/Tensordot/MatMul/pfor/ShapeShape1loop_body/Tensordot/Reshape/pfor/Reshape:output:0*
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
)loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape1loop_body/Tensordot/Reshape/pfor/Reshape:output:08loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:???????????????????
&loop_body/Tensordot/MatMul/pfor/MatMulMatMul2loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0&loop_body/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:?????????|
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
:??????????f
$loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
loop_body/Tensordot/pfor/concatConcatV2pfor/Reshape:output:0"loop_body/Tensordot/shape:output:0-loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
 loop_body/Tensordot/pfor/ReshapeReshape2loop_body/Tensordot/MatMul/pfor/Reshape_4:output:0(loop_body/Tensordot/pfor/concat:output:0*
T0*4
_output_shapes"
 :??????????r
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
,loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape)loop_body/Tensordot/pfor/Reshape:output:04loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:??????????(?
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
:??????????(?
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
:?????????~
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
T0*,
_output_shapes
:??????????(h
&loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
!loop_body/Tensordot_1/pfor/concatConcatV2pfor/Reshape:output:0$loop_body/Tensordot_1/shape:output:0/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
"loop_body/Tensordot_1/pfor/ReshapeReshape1loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:0*loop_body/Tensordot_1/pfor/concat:output:0*
T0*8
_output_shapes&
$:"??????????l
*loop_body/Tensordot_2/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
(loop_body/Tensordot_2/transpose/pfor/addAddV2-loop_body/Tensordot_2/transpose/perm:output:03loop_body/Tensordot_2/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:~
4loop_body/Tensordot_2/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: r
0loop_body/Tensordot_2/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+loop_body/Tensordot_2/transpose/pfor/concatConcatV2=loop_body/Tensordot_2/transpose/pfor/concat/values_0:output:0,loop_body/Tensordot_2/transpose/pfor/add:z:09loop_body/Tensordot_2/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
.loop_body/Tensordot_2/transpose/pfor/Transpose	Transpose+loop_body/Tensordot_1/pfor/Reshape:output:04loop_body/Tensordot_2/transpose/pfor/concat:output:0*
T0*8
_output_shapes&
$:"??????????r
0loop_body/Tensordot_2/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+loop_body/Tensordot_2/Reshape_1/pfor/concatConcatV2pfor/Reshape:output:0.loop_body/Tensordot_2/Reshape_1/shape:output:09loop_body/Tensordot_2/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,loop_body/Tensordot_2/Reshape_1/pfor/ReshapeReshape2loop_body/Tensordot_2/transpose/pfor/Transpose:y:04loop_body/Tensordot_2/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:??????????2?
'loop_body/Tensordot_2/MatMul/pfor/ShapeShape5loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
5loop_body/Tensordot_2/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/loop_body/Tensordot_2/MatMul/pfor/strided_sliceStridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0>loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_1StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
)loop_body/Tensordot_2/MatMul/pfor/MinimumMinimum8loop_body/Tensordot_2/MatMul/pfor/strided_slice:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: k
)loop_body/Tensordot_2/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :?
'loop_body/Tensordot_2/MatMul/pfor/EqualEqual-loop_body/Tensordot_2/MatMul/pfor/Minimum:z:02loop_body/Tensordot_2/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: 
*loop_body/Tensordot_2/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          
*loop_body/Tensordot_2/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          ?
(loop_body/Tensordot_2/MatMul/pfor/SelectSelect+loop_body/Tensordot_2/MatMul/pfor/Equal:z:03loop_body/Tensordot_2/MatMul/pfor/Select/t:output:03loop_body/Tensordot_2/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_2StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_3StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_4StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
'loop_body/Tensordot_2/MatMul/pfor/stackPack:loop_body/Tensordot_2/MatMul/pfor/strided_slice_2:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_3:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
+loop_body/Tensordot_2/MatMul/pfor/transpose	Transpose5loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:01loop_body/Tensordot_2/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'????????????????????????????
)loop_body/Tensordot_2/MatMul/pfor/ReshapeReshape/loop_body/Tensordot_2/MatMul/pfor/transpose:y:00loop_body/Tensordot_2/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:?2??????????
)loop_body/Tensordot_2/MatMul/pfor/Shape_1Shape2loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:s
1loop_body/Tensordot_2/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
'loop_body/Tensordot_2/MatMul/pfor/splitSplit:loop_body/Tensordot_2/MatMul/pfor/split/split_dim:output:02loop_body/Tensordot_2/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_splitt
1loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB v
3loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
+loop_body/Tensordot_2/MatMul/pfor/Reshape_1Reshape0loop_body/Tensordot_2/MatMul/pfor/split:output:0<loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: t
1loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB v
3loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
+loop_body/Tensordot_2/MatMul/pfor/Reshape_2Reshape0loop_body/Tensordot_2/MatMul/pfor/split:output:1<loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: t
1loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB v
3loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
+loop_body/Tensordot_2/MatMul/pfor/Reshape_3Reshape0loop_body/Tensordot_2/MatMul/pfor/split:output:2<loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ?
%loop_body/Tensordot_2/MatMul/pfor/mulMul4loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ?
1loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shapePack4loop_body/Tensordot_2/MatMul/pfor/Reshape_1:output:0)loop_body/Tensordot_2/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:?
+loop_body/Tensordot_2/MatMul/pfor/Reshape_4Reshape2loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0:loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:???????????????????
(loop_body/Tensordot_2/MatMul/pfor/MatMulMatMul&loop_body/Tensordot_2/Reshape:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:?????????~
3loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
??????????
1loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shapePack<loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:?
+loop_body/Tensordot_2/MatMul/pfor/Reshape_5Reshape2loop_body/Tensordot_2/MatMul/pfor/MatMul:product:0:loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'????????????????????????????
2loop_body/Tensordot_2/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
-loop_body/Tensordot_2/MatMul/pfor/transpose_1	Transpose4loop_body/Tensordot_2/MatMul/pfor/Reshape_5:output:0;loop_body/Tensordot_2/MatMul/pfor/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????h
&loop_body/Tensordot_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
!loop_body/Tensordot_2/pfor/concatConcatV2pfor/Reshape:output:0$loop_body/Tensordot_2/shape:output:0/loop_body/Tensordot_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
"loop_body/Tensordot_2/pfor/ReshapeReshape1loop_body/Tensordot_2/MatMul/pfor/transpose_1:y:0*loop_body/Tensordot_2/pfor/concat:output:0*
T0*/
_output_shapes
:?????????`
loop_body/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
loop_body/transpose/pfor/addAddV2!loop_body/transpose/perm:output:0'loop_body/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:r
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
:?
"loop_body/transpose/pfor/Transpose	Transpose+loop_body/Tensordot_2/pfor/Reshape:output:0(loop_body/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:?????????Y
loop_body/add/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :[
loop_body/add/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :Z
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
:?
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
:*
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
:?
loop_body/add/pfor/Reshape_1Reshape&loop_body/transpose/pfor/Transpose:y:0"loop_body/add/pfor/concat:output:0*
T0*/
_output_shapes
:??????????
loop_body/add/pfor/AddV2AddV2%loop_body/add/pfor/Reshape_1:output:0$loop_body/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   z
ReshapeReshapeloop_body/add/pfor/AddV2:z:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????@P
ReluReluReshape:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^loop_body/ReadVariableOp^loop_body/ReadVariableOp_1^loop_body/ReadVariableOp_2^loop_body/add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 24
loop_body/ReadVariableOploop_body/ReadVariableOp28
loop_body/ReadVariableOp_1loop_body/ReadVariableOp_128
loop_body/ReadVariableOp_2loop_body/ReadVariableOp_22<
loop_body/add/ReadVariableOploop_body/add/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_415593

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
?
H
1__inference_fruit_tn1_activity_regularizer_412340
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
?
?
*__inference_fruit_tn2_layer_call_fn_415719

inputs
unknown:$
	unknown_0:?
	unknown_1:
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_fruit_tn2_layer_call_and_return_conditional_losses_413174p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_1_layer_call_fn_415630

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
GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_412324?
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
?
J
3__inference_conv2dmpo_1_activity_regularizer_412315
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
?9
?
E__inference_conv2dmpo_layer_call_and_return_conditional_losses_412438

inputs1
readvariableop_resource:3
readvariableop_1_resource:/
!reshape_2_readvariableop_resource:
identity??ReadVariableOp?ReadVariableOp_1?Reshape_2/ReadVariableOpn
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0r
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*&
_output_shapes
:*
dtype0q
Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Tensordot/transpose	TransposeReadVariableOp_1:value:0!Tensordot/transpose/perm:output:0*
T0*&
_output_shapes
:h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
Tensordot/ReshapeReshapeTensordot/transpose:y:0 Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:s
Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Tensordot/transpose_1	TransposeReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*&
_output_shapes
:j
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:}
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*
_output_shapes

:p
Tensordot/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  ?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*.
_output_shapes
:o
transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   |
	transpose	TransposeTensordot:output:0transpose/perm:output:0*
T0*.
_output_shapes
:q
transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   {
transpose_1	Transposetranspose:y:0transpose_1/perm:output:0*
T0*.
_output_shapes
:f
ShapeConst*
_output_shapes
:*
dtype0*-
value$B""                  ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskO
ConstConst*
_output_shapes
:*
dtype0*
valueB: U
ProdProdstrided_slice:output:0Const:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskT
concat/values_1PackProd:output:0*
N*
T0*
_output_shapes
:V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2strided_slice_1:output:0concat/values_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapetranspose_1:y:0concat:output:0*
T0**
_output_shapes
:m
transpose_2/permConst*
_output_shapes
:*
dtype0*)
value B"                z
transpose_2	TransposeReshape:output:0transpose_2/perm:output:0*
T0**
_output_shapes
:d
Shape_1Const*
_output_shapes
:*
dtype0*)
value B"               _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskQ
Const_1Const*
_output_shapes
:*
dtype0*
valueB: [
Prod_1Prodstrided_slice_2:output:0Const_1:output:0*
T0*
_output_shapes
: _
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskX
concat_1/values_1PackProd_1:output:0*
N*
T0*
_output_shapes
:X
concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concat_1ConcatV2strided_slice_3:output:0concat_1/values_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:i
	Reshape_1Reshapetranspose_2:y:0concat_1:output:0*
T0*&
_output_shapes
:?
Conv2DConv2DinputsReshape_1:output:0*
T0*/
_output_shapes
:?????????//*
paddingSAME*
strides
v
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0`
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      y
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0*
_output_shapes

:k
addAddV2Conv2D:output:0Reshape_2:output:0*
T0*/
_output_shapes
:?????????//O
ReluReluadd:z:0*
T0*/
_output_shapes
:?????????//i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????//?
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^Reshape_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????//: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_124
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:W S
/
_output_shapes
:?????????//
 
_user_specified_nameinputs
?
?
*__inference_conv2dmpo_layer_call_fn_415570

inputs!
unknown:#
	unknown_0:
	unknown_1:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2dmpo_layer_call_and_return_conditional_losses_412438w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????//`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????//: : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????//
 
_user_specified_nameinputs
?	
c
D__inference_dropout1_layer_call_and_return_conditional_losses_413290

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?U
?
G__inference_conv2dmpo_1_layer_call_and_return_conditional_losses_412543

inputs1
readvariableop_resource:3
readvariableop_1_resource:3
readvariableop_2_resource:3
readvariableop_3_resource:0
!reshape_2_readvariableop_resource:	?
identity??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?Reshape_2/ReadVariableOpn
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0r
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*&
_output_shapes
:*
dtype0r
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*&
_output_shapes
:*
dtype0r
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*&
_output_shapes
:*
dtype0q
Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Tensordot/transpose	TransposeReadVariableOp_2:value:0!Tensordot/transpose/perm:output:0*
T0*&
_output_shapes
:h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@      ?
Tensordot/ReshapeReshapeTensordot/transpose:y:0 Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@s
Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Tensordot/transpose_1	TransposeReadVariableOp_3:value:0#Tensordot/transpose_1/perm:output:0*
T0*&
_output_shapes
:j
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:}
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*
_output_shapes

:@p
Tensordot/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  ?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*.
_output_shapes
:s
Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Tensordot_1/transpose	TransposeReadVariableOp:value:0#Tensordot_1/transpose/perm:output:0*
T0*&
_output_shapes
:j
Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
Tensordot_1/ReshapeReshapeTensordot_1/transpose:y:0"Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:u
Tensordot_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Tensordot_1/transpose_1	TransposeReadVariableOp_1:value:0%Tensordot_1/transpose_1/perm:output:0*
T0*&
_output_shapes
:l
Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   ?
Tensordot_1/Reshape_1ReshapeTensordot_1/transpose_1:y:0$Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:@?
Tensordot_1/MatMulMatMulTensordot_1/Reshape:output:0Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes

:@r
Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  ?
Tensordot_1ReshapeTensordot_1/MatMul:product:0Tensordot_1/shape:output:0*
T0*.
_output_shapes
:{
Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
Tensordot_2/transpose	TransposeTensordot:output:0#Tensordot_2/transpose/perm:output:0*
T0*.
_output_shapes
:j
Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?      ?
Tensordot_2/ReshapeReshapeTensordot_2/transpose:y:0"Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes
:	?}
Tensordot_2/transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
Tensordot_2/transpose_1	TransposeTensordot_1:output:0%Tensordot_2/transpose_1/perm:output:0*
T0*.
_output_shapes
:l
Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   ?
Tensordot_2/Reshape_1ReshapeTensordot_2/transpose_1:y:0$Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	??
Tensordot_2/MatMulMatMulTensordot_2/Reshape:output:0Tensordot_2/Reshape_1:output:0*
T0* 
_output_shapes
:
???
Tensordot_2/shapeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                              ?
Tensordot_2ReshapeTensordot_2/MatMul:product:0Tensordot_2/shape:output:0*
T0*>
_output_shapes,
*:(
transpose/permConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                      	         ?
	transpose	TransposeTensordot_2:output:0transpose/perm:output:0*
T0*>
_output_shapes,
*:(?
transpose_1/permConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                	               ?
transpose_1	Transposetranspose:y:0transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(v
ShapeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                              ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskO
ConstConst*
_output_shapes
:*
dtype0*
valueB: U
ProdProdstrided_slice:output:0Const:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskT
concat/values_1PackProd:output:0*
N*
T0*
_output_shapes
:V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2strided_slice_1:output:0concat/values_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:q
ReshapeReshapetranspose_1:y:0concat:output:0*
T0*2
_output_shapes 
:u
transpose_2/permConst*
_output_shapes
:*
dtype0*1
value(B&"                      ?
transpose_2	TransposeReshape:output:0transpose_2/perm:output:0*
T0*2
_output_shapes 
:l
Shape_1Const*
_output_shapes
:*
dtype0*1
value(B&"                     _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskQ
Const_1Const*
_output_shapes
:*
dtype0*
valueB: [
Prod_1Prodstrided_slice_2:output:0Const_1:output:0*
T0*
_output_shapes
: _
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskX
concat_1/values_1PackProd_1:output:0*
N*
T0*
_output_shapes
:X
concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concat_1ConcatV2strided_slice_3:output:0concat_1/values_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:j
	Reshape_1Reshapetranspose_2:y:0concat_1:output:0*
T0*'
_output_shapes
:??
Conv2DConv2DinputsReshape_1:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
w
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes	
:?*
dtype0`
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      z
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0*
_output_shapes
:	?l
addAddV2Conv2D:output:0Reshape_2:output:0*
T0*0
_output_shapes
:??????????P
ReluReluadd:z:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^Reshape_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_324
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
E__inference_fruit_tn1_layer_call_and_return_conditional_losses_416176

inputs7
!loop_body_readvariableop_resource:>
#loop_body_readvariableop_1_resource:?9
#loop_body_readvariableop_2_resource:;
%loop_body_add_readvariableop_resource:
identity??loop_body/ReadVariableOp?loop_body/ReadVariableOp_1?loop_body/ReadVariableOp_2?loop_body/add/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
:g
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
Tparams0*#
_output_shapes
:?~
loop_body/ReadVariableOpReadVariableOp!loop_body_readvariableop_resource*"
_output_shapes
:*
dtype0?
loop_body/ReadVariableOp_1ReadVariableOp#loop_body_readvariableop_1_resource*'
_output_shapes
:?*
dtype0?
loop_body/ReadVariableOp_2ReadVariableOp#loop_body_readvariableop_2_resource*"
_output_shapes
:*
dtype0w
"loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
loop_body/Tensordot/transpose	Transposeloop_body/GatherV2:output:0+loop_body/Tensordot/transpose/perm:output:0*
T0*#
_output_shapes
:?r
!loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
loop_body/Tensordot/ReshapeReshape!loop_body/Tensordot/transpose:y:0*loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	?y
$loop_body/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
loop_body/Tensordot/transpose_1	Transpose loop_body/ReadVariableOp:value:0-loop_body/Tensordot/transpose_1/perm:output:0*
T0*"
_output_shapes
:t
#loop_body/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
loop_body/Tensordot/Reshape_1Reshape#loop_body/Tensordot/transpose_1:y:0,loop_body/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:?
loop_body/Tensordot/MatMulMatMul$loop_body/Tensordot/Reshape:output:0&loop_body/Tensordot/Reshape_1:output:0*
T0*
_output_shapes
:	?r
loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            ?
loop_body/TensordotReshape$loop_body/Tensordot/MatMul:product:0"loop_body/Tensordot/shape:output:0*
T0*'
_output_shapes
:?y
$loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
loop_body/Tensordot_1/transpose	Transpose"loop_body/ReadVariableOp_2:value:0-loop_body/Tensordot_1/transpose/perm:output:0*
T0*"
_output_shapes
:t
#loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
loop_body/Tensordot_1/ReshapeReshape#loop_body/Tensordot_1/transpose:y:0,loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:v
%loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
loop_body/Tensordot_1/Reshape_1Reshapeloop_body/Tensordot:output:0.loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?(?
loop_body/Tensordot_1/MatMulMatMul&loop_body/Tensordot_1/Reshape:output:0(loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	?(x
loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"               ?
loop_body/Tensordot_1Reshape&loop_body/Tensordot_1/MatMul:product:0$loop_body/Tensordot_1/shape:output:0*
T0*+
_output_shapes
:?t
#loop_body/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
loop_body/Tensordot_2/ReshapeReshape"loop_body/ReadVariableOp_1:value:0,loop_body/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes
:	?2?
$loop_body/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
loop_body/Tensordot_2/transpose	Transposeloop_body/Tensordot_1:output:0-loop_body/Tensordot_2/transpose/perm:output:0*
T0*+
_output_shapes
:?v
%loop_body/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
loop_body/Tensordot_2/Reshape_1Reshape#loop_body/Tensordot_2/transpose:y:0.loop_body/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2?
loop_body/Tensordot_2/MatMulMatMul&loop_body/Tensordot_2/Reshape:output:0(loop_body/Tensordot_2/Reshape_1:output:0*
T0*
_output_shapes

:p
loop_body/Tensordot_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
loop_body/Tensordot_2Reshape&loop_body/Tensordot_2/MatMul:product:0$loop_body/Tensordot_2/shape:output:0*
T0*"
_output_shapes
:m
loop_body/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
loop_body/transpose	Transposeloop_body/Tensordot_2:output:0!loop_body/transpose/perm:output:0*
T0*"
_output_shapes
:?
loop_body/add/ReadVariableOpReadVariableOp%loop_body_add_readvariableop_resource*"
_output_shapes
:*
dtype0?
loop_body/addAddV2loop_body/transpose:y:0$loop_body/add/ReadVariableOp:value:0*
T0*"
_output_shapes
:\
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
Tparams0*0
_output_shapes
:??????????j
(loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
&loop_body/Tensordot/transpose/pfor/addAddV2+loop_body/Tensordot/transpose/perm:output:01loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:|
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
:?
,loop_body/Tensordot/transpose/pfor/Transpose	Transpose)loop_body/GatherV2/pfor/GatherV2:output:02loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*0
_output_shapes
:??????????n
,loop_body/Tensordot/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'loop_body/Tensordot/Reshape/pfor/concatConcatV2pfor/Reshape:output:0*loop_body/Tensordot/Reshape/shape:output:05loop_body/Tensordot/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
(loop_body/Tensordot/Reshape/pfor/ReshapeReshape0loop_body/Tensordot/transpose/pfor/Transpose:y:00loop_body/Tensordot/Reshape/pfor/concat:output:0*
T0*,
_output_shapes
:???????????
%loop_body/Tensordot/MatMul/pfor/ShapeShape1loop_body/Tensordot/Reshape/pfor/Reshape:output:0*
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
)loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape1loop_body/Tensordot/Reshape/pfor/Reshape:output:08loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:???????????????????
&loop_body/Tensordot/MatMul/pfor/MatMulMatMul2loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0&loop_body/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:?????????|
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
:??????????f
$loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
loop_body/Tensordot/pfor/concatConcatV2pfor/Reshape:output:0"loop_body/Tensordot/shape:output:0-loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
 loop_body/Tensordot/pfor/ReshapeReshape2loop_body/Tensordot/MatMul/pfor/Reshape_4:output:0(loop_body/Tensordot/pfor/concat:output:0*
T0*4
_output_shapes"
 :??????????r
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
,loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape)loop_body/Tensordot/pfor/Reshape:output:04loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:??????????(?
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
:??????????(?
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
:?????????~
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
T0*,
_output_shapes
:??????????(h
&loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
!loop_body/Tensordot_1/pfor/concatConcatV2pfor/Reshape:output:0$loop_body/Tensordot_1/shape:output:0/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
"loop_body/Tensordot_1/pfor/ReshapeReshape1loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:0*loop_body/Tensordot_1/pfor/concat:output:0*
T0*8
_output_shapes&
$:"??????????l
*loop_body/Tensordot_2/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
(loop_body/Tensordot_2/transpose/pfor/addAddV2-loop_body/Tensordot_2/transpose/perm:output:03loop_body/Tensordot_2/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:~
4loop_body/Tensordot_2/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: r
0loop_body/Tensordot_2/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+loop_body/Tensordot_2/transpose/pfor/concatConcatV2=loop_body/Tensordot_2/transpose/pfor/concat/values_0:output:0,loop_body/Tensordot_2/transpose/pfor/add:z:09loop_body/Tensordot_2/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
.loop_body/Tensordot_2/transpose/pfor/Transpose	Transpose+loop_body/Tensordot_1/pfor/Reshape:output:04loop_body/Tensordot_2/transpose/pfor/concat:output:0*
T0*8
_output_shapes&
$:"??????????r
0loop_body/Tensordot_2/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+loop_body/Tensordot_2/Reshape_1/pfor/concatConcatV2pfor/Reshape:output:0.loop_body/Tensordot_2/Reshape_1/shape:output:09loop_body/Tensordot_2/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,loop_body/Tensordot_2/Reshape_1/pfor/ReshapeReshape2loop_body/Tensordot_2/transpose/pfor/Transpose:y:04loop_body/Tensordot_2/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:??????????2?
'loop_body/Tensordot_2/MatMul/pfor/ShapeShape5loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
5loop_body/Tensordot_2/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/loop_body/Tensordot_2/MatMul/pfor/strided_sliceStridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0>loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_1StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
)loop_body/Tensordot_2/MatMul/pfor/MinimumMinimum8loop_body/Tensordot_2/MatMul/pfor/strided_slice:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: k
)loop_body/Tensordot_2/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :?
'loop_body/Tensordot_2/MatMul/pfor/EqualEqual-loop_body/Tensordot_2/MatMul/pfor/Minimum:z:02loop_body/Tensordot_2/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: 
*loop_body/Tensordot_2/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          
*loop_body/Tensordot_2/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          ?
(loop_body/Tensordot_2/MatMul/pfor/SelectSelect+loop_body/Tensordot_2/MatMul/pfor/Equal:z:03loop_body/Tensordot_2/MatMul/pfor/Select/t:output:03loop_body/Tensordot_2/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_2StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_3StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_4StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
'loop_body/Tensordot_2/MatMul/pfor/stackPack:loop_body/Tensordot_2/MatMul/pfor/strided_slice_2:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_3:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
+loop_body/Tensordot_2/MatMul/pfor/transpose	Transpose5loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:01loop_body/Tensordot_2/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'????????????????????????????
)loop_body/Tensordot_2/MatMul/pfor/ReshapeReshape/loop_body/Tensordot_2/MatMul/pfor/transpose:y:00loop_body/Tensordot_2/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:?2??????????
)loop_body/Tensordot_2/MatMul/pfor/Shape_1Shape2loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:s
1loop_body/Tensordot_2/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
'loop_body/Tensordot_2/MatMul/pfor/splitSplit:loop_body/Tensordot_2/MatMul/pfor/split/split_dim:output:02loop_body/Tensordot_2/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_splitt
1loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB v
3loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
+loop_body/Tensordot_2/MatMul/pfor/Reshape_1Reshape0loop_body/Tensordot_2/MatMul/pfor/split:output:0<loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: t
1loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB v
3loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
+loop_body/Tensordot_2/MatMul/pfor/Reshape_2Reshape0loop_body/Tensordot_2/MatMul/pfor/split:output:1<loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: t
1loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB v
3loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
+loop_body/Tensordot_2/MatMul/pfor/Reshape_3Reshape0loop_body/Tensordot_2/MatMul/pfor/split:output:2<loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ?
%loop_body/Tensordot_2/MatMul/pfor/mulMul4loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ?
1loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shapePack4loop_body/Tensordot_2/MatMul/pfor/Reshape_1:output:0)loop_body/Tensordot_2/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:?
+loop_body/Tensordot_2/MatMul/pfor/Reshape_4Reshape2loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0:loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:???????????????????
(loop_body/Tensordot_2/MatMul/pfor/MatMulMatMul&loop_body/Tensordot_2/Reshape:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:?????????~
3loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
??????????
1loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shapePack<loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:?
+loop_body/Tensordot_2/MatMul/pfor/Reshape_5Reshape2loop_body/Tensordot_2/MatMul/pfor/MatMul:product:0:loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'????????????????????????????
2loop_body/Tensordot_2/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
-loop_body/Tensordot_2/MatMul/pfor/transpose_1	Transpose4loop_body/Tensordot_2/MatMul/pfor/Reshape_5:output:0;loop_body/Tensordot_2/MatMul/pfor/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????h
&loop_body/Tensordot_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
!loop_body/Tensordot_2/pfor/concatConcatV2pfor/Reshape:output:0$loop_body/Tensordot_2/shape:output:0/loop_body/Tensordot_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:?
"loop_body/Tensordot_2/pfor/ReshapeReshape1loop_body/Tensordot_2/MatMul/pfor/transpose_1:y:0*loop_body/Tensordot_2/pfor/concat:output:0*
T0*/
_output_shapes
:?????????`
loop_body/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
loop_body/transpose/pfor/addAddV2!loop_body/transpose/perm:output:0'loop_body/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:r
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
:?
"loop_body/transpose/pfor/Transpose	Transpose+loop_body/Tensordot_2/pfor/Reshape:output:0(loop_body/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:?????????Y
loop_body/add/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :[
loop_body/add/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :Z
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
:?
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
:*
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
:?
loop_body/add/pfor/Reshape_1Reshape&loop_body/transpose/pfor/Transpose:y:0"loop_body/add/pfor/concat:output:0*
T0*/
_output_shapes
:??????????
loop_body/add/pfor/AddV2AddV2%loop_body/add/pfor/Reshape_1:output:0$loop_body/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   z
ReshapeReshapeloop_body/add/pfor/AddV2:z:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????@P
ReluReluReshape:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^loop_body/ReadVariableOp^loop_body/ReadVariableOp_1^loop_body/ReadVariableOp_2^loop_body/add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 24
loop_body/ReadVariableOploop_body/ReadVariableOp28
loop_body/ReadVariableOp_1loop_body/ReadVariableOp_128
loop_body/ReadVariableOp_2loop_body/ReadVariableOp_22<
loop_body/add/ReadVariableOploop_body/add/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_1_layer_call_fn_413240
input_12!
unknown:
	unknown_0:#
	unknown_1:#
	unknown_2:
	unknown_3:#
	unknown_4:#
	unknown_5:#
	unknown_6:#
	unknown_7:
	unknown_8:	?
	unknown_9:%

unknown_10:? 

unknown_11: 

unknown_12:

unknown_13:%

unknown_14:?

unknown_15:

unknown_16:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_12unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout	
2*
_collective_manager_ids
 *0
_output_shapes
:??????????: : : : *4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_413197p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????22: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????22
"
_user_specified_name
input_12
?
b
D__inference_dropout1_layer_call_and_return_conditional_losses_415686

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?_
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_413524

inputs'
conv2d_413443:
conv2d_413445:*
conv2dmpo_413448:*
conv2dmpo_413450:
conv2dmpo_413452:,
conv2dmpo_1_413464:,
conv2dmpo_1_413466:,
conv2dmpo_1_413468:,
conv2dmpo_1_413470:!
conv2dmpo_1_413472:	?&
fruit_tn1_413484:+
fruit_tn1_413486:?&
fruit_tn1_413488:&
fruit_tn1_413490:"
fruit_tn2_413502:+
fruit_tn2_413504:?"
fruit_tn2_413506:
fruit_tn2_413508:	?
identity

identity_1

identity_2

identity_3

identity_4??conv2d/StatefulPartitionedCall?!conv2dmpo/StatefulPartitionedCall?#conv2dmpo_1/StatefulPartitionedCall? dropout1/StatefulPartitionedCall?!fruit_tn1/StatefulPartitionedCall?!fruit_tn2/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_413443conv2d_413445*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_412370?
!conv2dmpo/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2dmpo_413448conv2dmpo_413450conv2dmpo_413452*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2dmpo_layer_call_and_return_conditional_losses_412438?
-conv2dmpo/ActivityRegularizer/PartitionedCallPartitionedCall*conv2dmpo/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *:
f5R3
1__inference_conv2dmpo_activity_regularizer_412290}
#conv2dmpo/ActivityRegularizer/ShapeShape*conv2dmpo/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1conv2dmpo/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3conv2dmpo/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3conv2dmpo/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+conv2dmpo/ActivityRegularizer/strided_sliceStridedSlice,conv2dmpo/ActivityRegularizer/Shape:output:0:conv2dmpo/ActivityRegularizer/strided_slice/stack:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_1:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"conv2dmpo/ActivityRegularizer/CastCast4conv2dmpo/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%conv2dmpo/ActivityRegularizer/truedivRealDiv6conv2dmpo/ActivityRegularizer/PartitionedCall:output:0&conv2dmpo/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
max_pooling2d/PartitionedCallPartitionedCall*conv2dmpo/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_412299?
#conv2dmpo_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2dmpo_1_413464conv2dmpo_1_413466conv2dmpo_1_413468conv2dmpo_1_413470conv2dmpo_1_413472*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2dmpo_1_layer_call_and_return_conditional_losses_412543?
/conv2dmpo_1/ActivityRegularizer/PartitionedCallPartitionedCall,conv2dmpo_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *<
f7R5
3__inference_conv2dmpo_1_activity_regularizer_412315?
%conv2dmpo_1/ActivityRegularizer/ShapeShape,conv2dmpo_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:}
3conv2dmpo_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5conv2dmpo_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5conv2dmpo_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-conv2dmpo_1/ActivityRegularizer/strided_sliceStridedSlice.conv2dmpo_1/ActivityRegularizer/Shape:output:0<conv2dmpo_1/ActivityRegularizer/strided_slice/stack:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_1:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
$conv2dmpo_1/ActivityRegularizer/CastCast6conv2dmpo_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
'conv2dmpo_1/ActivityRegularizer/truedivRealDiv8conv2dmpo_1/ActivityRegularizer/PartitionedCall:output:0(conv2dmpo_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
max_pooling2d_1/PartitionedCallPartitionedCall,conv2dmpo_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_412324?
!fruit_tn1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0fruit_tn1_413484fruit_tn1_413486fruit_tn1_413488fruit_tn1_413490*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_fruit_tn1_layer_call_and_return_conditional_losses_412856?
-fruit_tn1/ActivityRegularizer/PartitionedCallPartitionedCall*fruit_tn1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *:
f5R3
1__inference_fruit_tn1_activity_regularizer_412340}
#fruit_tn1/ActivityRegularizer/ShapeShape*fruit_tn1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1fruit_tn1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3fruit_tn1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3fruit_tn1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+fruit_tn1/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn1/ActivityRegularizer/Shape:output:0:fruit_tn1/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"fruit_tn1/ActivityRegularizer/CastCast4fruit_tn1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%fruit_tn1/ActivityRegularizer/truedivRealDiv6fruit_tn1/ActivityRegularizer/PartitionedCall:output:0&fruit_tn1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
 dropout1/StatefulPartitionedCallStatefulPartitionedCall*fruit_tn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout1_layer_call_and_return_conditional_losses_413290?
!fruit_tn2/StatefulPartitionedCallStatefulPartitionedCall)dropout1/StatefulPartitionedCall:output:0fruit_tn2_413502fruit_tn2_413504fruit_tn2_413506fruit_tn2_413508*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_fruit_tn2_layer_call_and_return_conditional_losses_413174?
-fruit_tn2/ActivityRegularizer/PartitionedCallPartitionedCall*fruit_tn2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *:
f5R3
1__inference_fruit_tn2_activity_regularizer_412353}
#fruit_tn2/ActivityRegularizer/ShapeShape*fruit_tn2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:{
1fruit_tn2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3fruit_tn2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3fruit_tn2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+fruit_tn2/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn2/ActivityRegularizer/Shape:output:0:fruit_tn2/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"fruit_tn2/ActivityRegularizer/CastCast4fruit_tn2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
%fruit_tn2/ActivityRegularizer/truedivRealDiv6fruit_tn2/ActivityRegularizer/PartitionedCall:output:0&fruit_tn2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: z
IdentityIdentity*fruit_tn2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????i

Identity_1Identity)conv2dmpo/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: k

Identity_2Identity+conv2dmpo_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_3Identity)fruit_tn1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: i

Identity_4Identity)fruit_tn2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^conv2d/StatefulPartitionedCall"^conv2dmpo/StatefulPartitionedCall$^conv2dmpo_1/StatefulPartitionedCall!^dropout1/StatefulPartitionedCall"^fruit_tn1/StatefulPartitionedCall"^fruit_tn2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????22: : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2F
!conv2dmpo/StatefulPartitionedCall!conv2dmpo/StatefulPartitionedCall2J
#conv2dmpo_1/StatefulPartitionedCall#conv2dmpo_1/StatefulPartitionedCall2D
 dropout1/StatefulPartitionedCall dropout1/StatefulPartitionedCall2F
!fruit_tn1/StatefulPartitionedCall!fruit_tn1/StatefulPartitionedCall2F
!fruit_tn2/StatefulPartitionedCall!fruit_tn2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????22
 
_user_specified_nameinputs
?

?
B__inference_conv2d_layer_call_and_return_conditional_losses_412370

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????//*
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
:?????????//g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????//w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????22
 
_user_specified_nameinputs
?	
c
D__inference_dropout1_layer_call_and_return_conditional_losses_415698

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
*__inference_fruit_tn1_layer_call_fn_415656

inputs
unknown:$
	unknown_0:?
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_fruit_tn1_layer_call_and_return_conditional_losses_412856o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_415635

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
?
?
-__inference_sequential_1_layer_call_fn_413612
input_12!
unknown:
	unknown_0:#
	unknown_1:#
	unknown_2:
	unknown_3:#
	unknown_4:#
	unknown_5:#
	unknown_6:#
	unknown_7:
	unknown_8:	?
	unknown_9:%

unknown_10:? 

unknown_11: 

unknown_12:

unknown_13:%

unknown_14:?

unknown_15:

unknown_16:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_12unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout	
2*
_collective_manager_ids
 *0
_output_shapes
:??????????: : : : *4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_413524p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????22: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????22
"
_user_specified_name
input_12"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_129
serving_default_input_12:0?????????22>
	fruit_tn21
StatefulPartitionedCall:0??????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	nodes
end_node_first
end_node_last
bias
bias_var
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	*nodes
+end_node_first
,middle_node_0
-middle_node_1
.end_node_last
/bias
/bias_var
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
?
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	<a_var
	=b_var
	>c_var
?bias
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses"
_tf_keras_layer
?
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J_random_generator
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	Ma_var
	Nb_var
	Oc_var
Pbias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Witer

Xbeta_1

Ybeta_2
	Zdecaym?m?m?m?m?+m?,m?-m?.m?/m?<m?=m?>m??m?Mm?Nm?Om?Pm?v?v?v?v?v?+v?,v?-v?.v?/v?<v?=v?>v??v?Mv?Nv?Ov?Pv?vhat?vhat?vhat?vhat?vhat?+vhat?,vhat?-vhat?.vhat?/vhat?<vhat?=vhat?>vhat??vhat?Mvhat?Nvhat?Ovhat?Pvhat?"
	optimizer
?
0
1
2
3
4
+5
,6
-7
.8
/9
<10
=11
>12
?13
M14
N15
O16
P17"
trackable_list_wrapper
?
0
1
2
3
4
+5
,6
-7
.8
/9
<10
=11
>12
?13
M14
N15
O16
P17"
trackable_list_wrapper
 "
trackable_list_wrapper
?
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_sequential_1_layer_call_fn_413240
-__inference_sequential_1_layer_call_fn_413861
-__inference_sequential_1_layer_call_fn_413906
-__inference_sequential_1_layer_call_fn_413612?
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
H__inference_sequential_1_layer_call_and_return_conditional_losses_414698
H__inference_sequential_1_layer_call_and_return_conditional_losses_415497
H__inference_sequential_1_layer_call_and_return_conditional_losses_413696
H__inference_sequential_1_layer_call_and_return_conditional_losses_413780?
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
!__inference__wrapped_model_412277input_12"?
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
`serving_default"
signature_map
':%2conv2d/kernel
:2conv2d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_conv2d_layer_call_fn_415549?
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
B__inference_conv2d_layer_call_and_return_conditional_losses_415559?
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
.
0
1"
trackable_list_wrapper
2:02conv2dmpo/end_node_first
1:/2conv2dmpo/end_node_last
:2conv2dmpo/bias
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
kactivity_regularizer_fn
*#&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_conv2dmpo_layer_call_fn_415570?
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
I__inference_conv2dmpo_layer_call_and_return_all_conditional_losses_415583?
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
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
?2?
.__inference_max_pooling2d_layer_call_fn_415588?
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
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_415593?
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
<
+0
,1
-2
.3"
trackable_list_wrapper
4:22conv2dmpo_1/end_node_first
3:12conv2dmpo_1/middle_node_0
3:12conv2dmpo_1/middle_node_1
3:12conv2dmpo_1/end_node_last
:?2conv2dmpo_1/bias
C
+0
,1
-2
.3
/4"
trackable_list_wrapper
C
+0
,1
-2
.3
/4"
trackable_list_wrapper
 "
trackable_list_wrapper
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
wactivity_regularizer_fn
*5&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_conv2dmpo_1_layer_call_fn_415608?
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
K__inference_conv2dmpo_1_layer_call_and_return_all_conditional_losses_415625?
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
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
?2?
0__inference_max_pooling2d_1_layer_call_fn_415630?
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
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_415635?
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
:2a
:?2b
:2c
:2bias
<
<0
=1
>2
?3"
trackable_list_wrapper
<
<0
=1
>2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
~non_trainable_variables

layers
?metrics
 ?layer_regularization_losses
?layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
?activity_regularizer_fn
*E&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_fruit_tn1_layer_call_fn_415656?
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
I__inference_fruit_tn1_layer_call_and_return_all_conditional_losses_415671?
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
)__inference_dropout1_layer_call_fn_415676
)__inference_dropout1_layer_call_fn_415681?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout1_layer_call_and_return_conditional_losses_415686
D__inference_dropout1_layer_call_and_return_conditional_losses_415698?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:2a
:?2b
:2c
:?2bias
<
M0
N1
O2
P3"
trackable_list_wrapper
<
M0
N1
O2
P3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
?activity_regularizer_fn
*V&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_fruit_tn2_layer_call_fn_415719?
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
I__inference_fruit_tn2_layer_call_and_return_all_conditional_losses_415734?
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
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
$__inference_signature_wrapper_415540input_12"?
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
1__inference_conv2dmpo_activity_regularizer_412290?
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
E__inference_conv2dmpo_layer_call_and_return_conditional_losses_415796?
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
?2?
3__inference_conv2dmpo_1_activity_regularizer_412315?
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
G__inference_conv2dmpo_1_layer_call_and_return_conditional_losses_415884?
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
?2?
1__inference_fruit_tn1_activity_regularizer_412340?
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
E__inference_fruit_tn1_layer_call_and_return_conditional_losses_416176?
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
?2?
1__inference_fruit_tn2_activity_regularizer_412353?
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
E__inference_fruit_tn2_layer_call_and_return_conditional_losses_416469?
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
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
v
?
thresholds
?true_positives
?false_positives
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
7:52Adam/conv2dmpo/end_node_first/m
6:42Adam/conv2dmpo/end_node_last/m
!:2Adam/conv2dmpo/bias/m
9:72!Adam/conv2dmpo_1/end_node_first/m
8:62 Adam/conv2dmpo_1/middle_node_0/m
8:62 Adam/conv2dmpo_1/middle_node_1/m
8:62 Adam/conv2dmpo_1/end_node_last/m
$:"?2Adam/conv2dmpo_1/bias/m
:2Adam/a/m
!:?2Adam/b/m
:2Adam/c/m
:2Adam/bias/m
:2Adam/a/m
!:?2Adam/b/m
:2Adam/c/m
:?2Adam/bias/m
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
7:52Adam/conv2dmpo/end_node_first/v
6:42Adam/conv2dmpo/end_node_last/v
!:2Adam/conv2dmpo/bias/v
9:72!Adam/conv2dmpo_1/end_node_first/v
8:62 Adam/conv2dmpo_1/middle_node_0/v
8:62 Adam/conv2dmpo_1/middle_node_1/v
8:62 Adam/conv2dmpo_1/end_node_last/v
$:"?2Adam/conv2dmpo_1/bias/v
:2Adam/a/v
!:?2Adam/b/v
:2Adam/c/v
:2Adam/bias/v
:2Adam/a/v
!:?2Adam/b/v
:2Adam/c/v
:?2Adam/bias/v
/:-2Adam/conv2d/kernel/vhat
!:2Adam/conv2d/bias/vhat
::82"Adam/conv2dmpo/end_node_first/vhat
9:72!Adam/conv2dmpo/end_node_last/vhat
$:"2Adam/conv2dmpo/bias/vhat
<::2$Adam/conv2dmpo_1/end_node_first/vhat
;:92#Adam/conv2dmpo_1/middle_node_0/vhat
;:92#Adam/conv2dmpo_1/middle_node_1/vhat
;:92#Adam/conv2dmpo_1/end_node_last/vhat
':%?2Adam/conv2dmpo_1/bias/vhat
:2Adam/a/vhat
$:"?2Adam/b/vhat
:2Adam/c/vhat
": 2Adam/bias/vhat
:2Adam/a/vhat
$:"?2Adam/b/vhat
:2Adam/c/vhat
:?2Adam/bias/vhat?
!__inference__wrapped_model_412277?+,-./<=>?MNOP9?6
/?,
*?'
input_12?????????22
? "6?3
1
	fruit_tn2$?!
	fruit_tn2???????????
B__inference_conv2d_layer_call_and_return_conditional_losses_415559l7?4
-?*
(?%
inputs?????????22
? "-?*
#? 
0?????????//
? ?
'__inference_conv2d_layer_call_fn_415549_7?4
-?*
(?%
inputs?????????22
? " ??????????//]
3__inference_conv2dmpo_1_activity_regularizer_412315&?
?
?	
x
? "? ?
K__inference_conv2dmpo_1_layer_call_and_return_all_conditional_losses_415625~+,-./7?4
-?*
(?%
inputs?????????
? "<?9
$?!
0??????????
?
?	
1/0 ?
G__inference_conv2dmpo_1_layer_call_and_return_conditional_losses_415884p+,-./7?4
-?*
(?%
inputs?????????
? ".?+
$?!
0??????????
? ?
,__inference_conv2dmpo_1_layer_call_fn_415608c+,-./7?4
-?*
(?%
inputs?????????
? "!???????????[
1__inference_conv2dmpo_activity_regularizer_412290&?
?
?	
x
? "? ?
I__inference_conv2dmpo_layer_call_and_return_all_conditional_losses_415583{7?4
-?*
(?%
inputs?????????//
? ";?8
#? 
0?????????//
?
?	
1/0 ?
E__inference_conv2dmpo_layer_call_and_return_conditional_losses_415796m7?4
-?*
(?%
inputs?????????//
? "-?*
#? 
0?????????//
? ?
*__inference_conv2dmpo_layer_call_fn_415570`7?4
-?*
(?%
inputs?????????//
? " ??????????//?
D__inference_dropout1_layer_call_and_return_conditional_losses_415686\3?0
)?&
 ?
inputs?????????@
p 
? "%?"
?
0?????????@
? ?
D__inference_dropout1_layer_call_and_return_conditional_losses_415698\3?0
)?&
 ?
inputs?????????@
p
? "%?"
?
0?????????@
? |
)__inference_dropout1_layer_call_fn_415676O3?0
)?&
 ?
inputs?????????@
p 
? "??????????@|
)__inference_dropout1_layer_call_fn_415681O3?0
)?&
 ?
inputs?????????@
p
? "??????????@[
1__inference_fruit_tn1_activity_regularizer_412340&?
?
?	
x
? "? ?
I__inference_fruit_tn1_layer_call_and_return_all_conditional_losses_415671u<=>?8?5
.?+
)?&
inputs??????????
? "3?0
?
0?????????@
?
?	
1/0 ?
E__inference_fruit_tn1_layer_call_and_return_conditional_losses_416176g<=>?8?5
.?+
)?&
inputs??????????
? "%?"
?
0?????????@
? ?
*__inference_fruit_tn1_layer_call_fn_415656Z<=>?8?5
.?+
)?&
inputs??????????
? "??????????@[
1__inference_fruit_tn2_activity_regularizer_412353&?
?
?	
x
? "? ?
I__inference_fruit_tn2_layer_call_and_return_all_conditional_losses_415734mMNOP/?,
%?"
 ?
inputs?????????@
? "4?1
?
0??????????
?
?	
1/0 ?
E__inference_fruit_tn2_layer_call_and_return_conditional_losses_416469_MNOP/?,
%?"
 ?
inputs?????????@
? "&?#
?
0??????????
? ?
*__inference_fruit_tn2_layer_call_fn_415719RMNOP/?,
%?"
 ?
inputs?????????@
? "????????????
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_415635?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_1_layer_call_fn_415630?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_415593?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_max_pooling2d_layer_call_fn_415588?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
H__inference_sequential_1_layer_call_and_return_conditional_losses_413696?+,-./<=>?MNOPA?>
7?4
*?'
input_12?????????22
p 

 
? "^?[
?
0??????????
;?8
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_413780?+,-./<=>?MNOPA?>
7?4
*?'
input_12?????????22
p

 
? "^?[
?
0??????????
;?8
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_414698?+,-./<=>?MNOP??<
5?2
(?%
inputs?????????22
p 

 
? "^?[
?
0??????????
;?8
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_415497?+,-./<=>?MNOP??<
5?2
(?%
inputs?????????22
p

 
? "^?[
?
0??????????
;?8
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 ?
-__inference_sequential_1_layer_call_fn_413240r+,-./<=>?MNOPA?>
7?4
*?'
input_12?????????22
p 

 
? "????????????
-__inference_sequential_1_layer_call_fn_413612r+,-./<=>?MNOPA?>
7?4
*?'
input_12?????????22
p

 
? "????????????
-__inference_sequential_1_layer_call_fn_413861p+,-./<=>?MNOP??<
5?2
(?%
inputs?????????22
p 

 
? "????????????
-__inference_sequential_1_layer_call_fn_413906p+,-./<=>?MNOP??<
5?2
(?%
inputs?????????22
p

 
? "????????????
$__inference_signature_wrapper_415540?+,-./<=>?MNOPE?B
? 
;?8
6
input_12*?'
input_12?????????22"6?3
1
	fruit_tn2$?!
	fruit_tn2??????????