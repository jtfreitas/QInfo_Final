Џч4
Џ$Ђ$
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
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
Ы
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
incompatible_shape_errorbool(Р
≠
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
М
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
В
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
delete_old_dirsbool(И
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
2	Р
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
Н
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
Ѕ
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
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
ц
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
М
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68їЖ3
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
Ф
conv2dmpo/end_node_firstVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2dmpo/end_node_first
Н
,conv2dmpo/end_node_first/Read/ReadVariableOpReadVariableOpconv2dmpo/end_node_first*&
_output_shapes
:*
dtype0
Т
conv2dmpo/end_node_lastVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2dmpo/end_node_last
Л
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
Ш
conv2dmpo_1/end_node_firstVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv2dmpo_1/end_node_first
С
.conv2dmpo_1/end_node_first/Read/ReadVariableOpReadVariableOpconv2dmpo_1/end_node_first*&
_output_shapes
:*
dtype0
Ц
conv2dmpo_1/middle_node_0VarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2dmpo_1/middle_node_0
П
-conv2dmpo_1/middle_node_0/Read/ReadVariableOpReadVariableOpconv2dmpo_1/middle_node_0*&
_output_shapes
:*
dtype0
Ц
conv2dmpo_1/middle_node_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2dmpo_1/middle_node_1
П
-conv2dmpo_1/middle_node_1/Read/ReadVariableOpReadVariableOpconv2dmpo_1/middle_node_1*&
_output_shapes
:*
dtype0
Ц
conv2dmpo_1/end_node_lastVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2dmpo_1/end_node_last
П
-conv2dmpo_1/end_node_last/Read/ReadVariableOpReadVariableOpconv2dmpo_1/end_node_last*&
_output_shapes
:*
dtype0
y
conv2dmpo_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*!
shared_nameconv2dmpo_1/bias
r
$conv2dmpo_1/bias/Read/ReadVariableOpReadVariableOpconv2dmpo_1/bias*
_output_shapes	
:А*
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
shape:А*
shared_nameb
`
b/Read/ReadVariableOpReadVariableOpb*'
_output_shapes
:А*
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
shape:Г*
shared_nameb_1
d
b_1/Read/ReadVariableOpReadVariableOpb_1*'
_output_shapes
:Г*
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
shape:Г*
shared_namebias_1
^
bias_1/Read/ReadVariableOpReadVariableOpbias_1*
_output_shapes	
:Г*
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
М
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m
Е
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
Ґ
Adam/conv2dmpo/end_node_first/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2dmpo/end_node_first/m
Ы
3Adam/conv2dmpo/end_node_first/m/Read/ReadVariableOpReadVariableOpAdam/conv2dmpo/end_node_first/m*&
_output_shapes
:*
dtype0
†
Adam/conv2dmpo/end_node_last/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2dmpo/end_node_last/m
Щ
2Adam/conv2dmpo/end_node_last/m/Read/ReadVariableOpReadVariableOpAdam/conv2dmpo/end_node_last/m*&
_output_shapes
:*
dtype0
В
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
¶
!Adam/conv2dmpo_1/end_node_first/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2dmpo_1/end_node_first/m
Я
5Adam/conv2dmpo_1/end_node_first/m/Read/ReadVariableOpReadVariableOp!Adam/conv2dmpo_1/end_node_first/m*&
_output_shapes
:*
dtype0
§
 Adam/conv2dmpo_1/middle_node_0/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2dmpo_1/middle_node_0/m
Э
4Adam/conv2dmpo_1/middle_node_0/m/Read/ReadVariableOpReadVariableOp Adam/conv2dmpo_1/middle_node_0/m*&
_output_shapes
:*
dtype0
§
 Adam/conv2dmpo_1/middle_node_1/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2dmpo_1/middle_node_1/m
Э
4Adam/conv2dmpo_1/middle_node_1/m/Read/ReadVariableOpReadVariableOp Adam/conv2dmpo_1/middle_node_1/m*&
_output_shapes
:*
dtype0
§
 Adam/conv2dmpo_1/end_node_last/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2dmpo_1/end_node_last/m
Э
4Adam/conv2dmpo_1/end_node_last/m/Read/ReadVariableOpReadVariableOp Adam/conv2dmpo_1/end_node_last/m*&
_output_shapes
:*
dtype0
З
Adam/conv2dmpo_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameAdam/conv2dmpo_1/bias/m
А
+Adam/conv2dmpo_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2dmpo_1/bias/m*
_output_shapes	
:А*
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
shape:А*
shared_name
Adam/b/m
n
Adam/b/m/Read/ReadVariableOpReadVariableOpAdam/b/m*'
_output_shapes
:А*
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
shape:Г*
shared_name
Adam/b/m_1
r
Adam/b/m_1/Read/ReadVariableOpReadVariableOp
Adam/b/m_1*'
_output_shapes
:Г*
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
shape:Г*
shared_nameAdam/bias/m_1
l
!Adam/bias/m_1/Read/ReadVariableOpReadVariableOpAdam/bias/m_1*
_output_shapes	
:Г*
dtype0
М
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v
Е
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
Ґ
Adam/conv2dmpo/end_node_first/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2dmpo/end_node_first/v
Ы
3Adam/conv2dmpo/end_node_first/v/Read/ReadVariableOpReadVariableOpAdam/conv2dmpo/end_node_first/v*&
_output_shapes
:*
dtype0
†
Adam/conv2dmpo/end_node_last/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2dmpo/end_node_last/v
Щ
2Adam/conv2dmpo/end_node_last/v/Read/ReadVariableOpReadVariableOpAdam/conv2dmpo/end_node_last/v*&
_output_shapes
:*
dtype0
В
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
¶
!Adam/conv2dmpo_1/end_node_first/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2dmpo_1/end_node_first/v
Я
5Adam/conv2dmpo_1/end_node_first/v/Read/ReadVariableOpReadVariableOp!Adam/conv2dmpo_1/end_node_first/v*&
_output_shapes
:*
dtype0
§
 Adam/conv2dmpo_1/middle_node_0/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2dmpo_1/middle_node_0/v
Э
4Adam/conv2dmpo_1/middle_node_0/v/Read/ReadVariableOpReadVariableOp Adam/conv2dmpo_1/middle_node_0/v*&
_output_shapes
:*
dtype0
§
 Adam/conv2dmpo_1/middle_node_1/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2dmpo_1/middle_node_1/v
Э
4Adam/conv2dmpo_1/middle_node_1/v/Read/ReadVariableOpReadVariableOp Adam/conv2dmpo_1/middle_node_1/v*&
_output_shapes
:*
dtype0
§
 Adam/conv2dmpo_1/end_node_last/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2dmpo_1/end_node_last/v
Э
4Adam/conv2dmpo_1/end_node_last/v/Read/ReadVariableOpReadVariableOp Adam/conv2dmpo_1/end_node_last/v*&
_output_shapes
:*
dtype0
З
Adam/conv2dmpo_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameAdam/conv2dmpo_1/bias/v
А
+Adam/conv2dmpo_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2dmpo_1/bias/v*
_output_shapes	
:А*
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
shape:А*
shared_name
Adam/b/v
n
Adam/b/v/Read/ReadVariableOpReadVariableOpAdam/b/v*'
_output_shapes
:А*
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
shape:Г*
shared_name
Adam/b/v_1
r
Adam/b/v_1/Read/ReadVariableOpReadVariableOp
Adam/b/v_1*'
_output_shapes
:Г*
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
shape:Г*
shared_nameAdam/bias/v_1
l
!Adam/bias/v_1/Read/ReadVariableOpReadVariableOpAdam/bias/v_1*
_output_shapes	
:Г*
dtype0
Т
Adam/conv2d/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d/kernel/vhat
Л
+Adam/conv2d/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/vhat*&
_output_shapes
:*
dtype0
В
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
®
"Adam/conv2dmpo/end_node_first/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/conv2dmpo/end_node_first/vhat
°
6Adam/conv2dmpo/end_node_first/vhat/Read/ReadVariableOpReadVariableOp"Adam/conv2dmpo/end_node_first/vhat*&
_output_shapes
:*
dtype0
¶
!Adam/conv2dmpo/end_node_last/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2dmpo/end_node_last/vhat
Я
5Adam/conv2dmpo/end_node_last/vhat/Read/ReadVariableOpReadVariableOp!Adam/conv2dmpo/end_node_last/vhat*&
_output_shapes
:*
dtype0
И
Adam/conv2dmpo/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2dmpo/bias/vhat
Б
,Adam/conv2dmpo/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/conv2dmpo/bias/vhat*
_output_shapes
:*
dtype0
ђ
$Adam/conv2dmpo_1/end_node_first/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/conv2dmpo_1/end_node_first/vhat
•
8Adam/conv2dmpo_1/end_node_first/vhat/Read/ReadVariableOpReadVariableOp$Adam/conv2dmpo_1/end_node_first/vhat*&
_output_shapes
:*
dtype0
™
#Adam/conv2dmpo_1/middle_node_0/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/conv2dmpo_1/middle_node_0/vhat
£
7Adam/conv2dmpo_1/middle_node_0/vhat/Read/ReadVariableOpReadVariableOp#Adam/conv2dmpo_1/middle_node_0/vhat*&
_output_shapes
:*
dtype0
™
#Adam/conv2dmpo_1/middle_node_1/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/conv2dmpo_1/middle_node_1/vhat
£
7Adam/conv2dmpo_1/middle_node_1/vhat/Read/ReadVariableOpReadVariableOp#Adam/conv2dmpo_1/middle_node_1/vhat*&
_output_shapes
:*
dtype0
™
#Adam/conv2dmpo_1/end_node_last/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/conv2dmpo_1/end_node_last/vhat
£
7Adam/conv2dmpo_1/end_node_last/vhat/Read/ReadVariableOpReadVariableOp#Adam/conv2dmpo_1/end_node_last/vhat*&
_output_shapes
:*
dtype0
Н
Adam/conv2dmpo_1/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameAdam/conv2dmpo_1/bias/vhat
Ж
.Adam/conv2dmpo_1/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/conv2dmpo_1/bias/vhat*
_output_shapes	
:А*
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
shape:А*
shared_nameAdam/b/vhat
t
Adam/b/vhat/Read/ReadVariableOpReadVariableOpAdam/b/vhat*'
_output_shapes
:А*
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
shape:Г*
shared_nameAdam/b/vhat_1
x
!Adam/b/vhat_1/Read/ReadVariableOpReadVariableOpAdam/b/vhat_1*'
_output_shapes
:Г*
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
shape:Г*!
shared_nameAdam/bias/vhat_1
r
$Adam/bias/vhat_1/Read/ReadVariableOpReadVariableOpAdam/bias/vhat_1*
_output_shapes	
:Г*
dtype0

NoOpNoOp
ќ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Й
value€~Bь~ Bх~
ґ
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
¶

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
Џ
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
О
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses* 
А
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
О
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses* 
ї
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
•
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J_random_generator
K__call__
*L&call_and_return_all_conditional_losses* 
ї
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
Г
Witer

Xbeta_1

Ybeta_2
	ZdecaymЬmЭmЮmЯm†+m°,mҐ-m£.m§/m•<m¶=mІ>m®?m©Mm™NmЂOmђPm≠vЃvѓv∞v±v≤+v≥,vі-vµ.vґ/vЈ<vЄ=vє>vЇ?vїMvЉNvљOvЊPvњvhatјvhatЅvhat¬vhat√vhatƒ+vhat≈,vhat∆-vhat«.vhat»/vhat…<vhat =vhatЋ>vhatћ?vhatЌMvhatќNvhatѕOvhat–Pvhat—*
К
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
К
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
∞
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
У
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
∞
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
С
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
∞
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
С
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
µ
~non_trainable_variables

layers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
Гactivity_regularizer_fn
*E&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ц
Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
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
Ј
Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
Пactivity_regularizer_fn
*V&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses*
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
С0
Т1*
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

Уtotal

Фcount
Х	variables
Ц	keras_api*
`
Ч
thresholds
Шtrue_positives
Щfalse_positives
Ъ	variables
Ы	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

У0
Ф1*

Х	variables*
* 
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

Ш0
Щ1*

Ъ	variables*
Аz
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ФН
VARIABLE_VALUEAdam/conv2dmpo/end_node_first/mZlayer_with_weights-1/end_node_first/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ТЛ
VARIABLE_VALUEAdam/conv2dmpo/end_node_last/mYlayer_with_weights-1/end_node_last/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2dmpo/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЦП
VARIABLE_VALUE!Adam/conv2dmpo_1/end_node_first/mZlayer_with_weights-2/end_node_first/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ФН
VARIABLE_VALUE Adam/conv2dmpo_1/middle_node_0/mYlayer_with_weights-2/middle_node_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ФН
VARIABLE_VALUE Adam/conv2dmpo_1/middle_node_1/mYlayer_with_weights-2/middle_node_1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ФН
VARIABLE_VALUE Adam/conv2dmpo_1/end_node_last/mYlayer_with_weights-2/end_node_last/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
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
Аz
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ФН
VARIABLE_VALUEAdam/conv2dmpo/end_node_first/vZlayer_with_weights-1/end_node_first/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ТЛ
VARIABLE_VALUEAdam/conv2dmpo/end_node_last/vYlayer_with_weights-1/end_node_last/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2dmpo/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЦП
VARIABLE_VALUE!Adam/conv2dmpo_1/end_node_first/vZlayer_with_weights-2/end_node_first/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ФН
VARIABLE_VALUE Adam/conv2dmpo_1/middle_node_0/vYlayer_with_weights-2/middle_node_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ФН
VARIABLE_VALUE Adam/conv2dmpo_1/middle_node_1/vYlayer_with_weights-2/middle_node_1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ФН
VARIABLE_VALUE Adam/conv2dmpo_1/end_node_last/vYlayer_with_weights-2/end_node_last/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
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
ЗА
VARIABLE_VALUEAdam/conv2d/kernel/vhatUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d/bias/vhatSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
ЪУ
VARIABLE_VALUE"Adam/conv2dmpo/end_node_first/vhat]layer_with_weights-1/end_node_first/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
ШС
VARIABLE_VALUE!Adam/conv2dmpo/end_node_last/vhat\layer_with_weights-1/end_node_last/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUEAdam/conv2dmpo/bias/vhatSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
ЬХ
VARIABLE_VALUE$Adam/conv2dmpo_1/end_node_first/vhat]layer_with_weights-2/end_node_first/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
ЪУ
VARIABLE_VALUE#Adam/conv2dmpo_1/middle_node_0/vhat\layer_with_weights-2/middle_node_0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
ЪУ
VARIABLE_VALUE#Adam/conv2dmpo_1/middle_node_1/vhat\layer_with_weights-2/middle_node_1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
ЪУ
VARIABLE_VALUE#Adam/conv2dmpo_1/end_node_last/vhat\layer_with_weights-2/end_node_last/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
ИБ
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
Л
serving_default_input_12Placeholder*/
_output_shapes
:€€€€€€€€€22*
dtype0*$
shape:€€€€€€€€€22
я
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_12conv2d/kernelconv2d/biasconv2dmpo/end_node_firstconv2dmpo/end_node_lastconv2dmpo/biasconv2dmpo_1/end_node_firstconv2dmpo_1/middle_node_0conv2dmpo_1/middle_node_1conv2dmpo_1/end_node_lastconv2dmpo_1/biasabcbiasa_1b_1c_1bias_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Г*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_415540
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ќ
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
GPU2*0J 8В *(
f#R!
__inference__traced_save_416732
Й
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
GPU2*0J 8В *+
f&R$
"__inference__traced_restore_416982Ті0
їз
Ї
E__inference_fruit_tn2_layer_call_and_return_conditional_losses_416469

inputs3
!loop_body_readvariableop_resource:>
#loop_body_readvariableop_1_resource:Г5
#loop_body_readvariableop_2_resource:4
%loop_body_add_readvariableop_resource:	Г
identityИҐloop_body/ReadVariableOpҐloop_body/ReadVariableOp_1Ґloop_body/ReadVariableOp_2Ґloop_body/add/ReadVariableOp;
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
valueB:—
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
value	B : Э
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
valueB:Г
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
value	B : †
loop_body/SelectV2SelectV2loop_body/Greater:z:0)loop_body/PlaceholderWithDefault:output:0loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: Y
loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ≠
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
valueB"         И
loop_body/ReshapeReshapeloop_body/GatherV2:output:0 loop_body/Reshape/shape:output:0*
T0*"
_output_shapes
:z
loop_body/ReadVariableOpReadVariableOp!loop_body_readvariableop_resource*
_output_shapes

:*
dtype0З
loop_body/ReadVariableOp_1ReadVariableOp#loop_body_readvariableop_1_resource*'
_output_shapes
:Г*
dtype0~
loop_body/ReadVariableOp_2ReadVariableOp#loop_body_readvariableop_2_resource*
_output_shapes

:*
dtype0w
"loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          †
loop_body/Tensordot/transpose	Transposeloop_body/Reshape:output:0+loop_body/Tensordot/transpose/perm:output:0*
T0*"
_output_shapes
:r
!loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ю
loop_body/Tensordot/ReshapeReshape!loop_body/Tensordot/transpose:y:0*loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:Х
loop_body/Tensordot/MatMulMatMul$loop_body/Tensordot/Reshape:output:0 loop_body/ReadVariableOp:value:0*
T0*
_output_shapes

:n
loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         Х
loop_body/TensordotReshape$loop_body/Tensordot/MatMul:product:0"loop_body/Tensordot/shape:output:0*
T0*"
_output_shapes
:}
$loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ±
loop_body/Tensordot_1/transpose	Transpose"loop_body/ReadVariableOp_1:value:0-loop_body/Tensordot_1/transpose/perm:output:0*
T0*'
_output_shapes
:Гt
#loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"     •
loop_body/Tensordot_1/ReshapeReshape#loop_body/Tensordot_1/transpose:y:0,loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes
:	М{
&loop_body/Tensordot_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ™
!loop_body/Tensordot_1/transpose_1	Transposeloop_body/Tensordot:output:0/loop_body/Tensordot_1/transpose_1/perm:output:0*
T0*"
_output_shapes
:v
%loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ™
loop_body/Tensordot_1/Reshape_1Reshape%loop_body/Tensordot_1/transpose_1:y:0.loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:Ґ
loop_body/Tensordot_1/MatMulMatMul&loop_body/Tensordot_1/Reshape:output:0(loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	Мp
loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"Г         Ь
loop_body/Tensordot_1Reshape&loop_body/Tensordot_1/MatMul:product:0$loop_body/Tensordot_1/shape:output:0*
T0*#
_output_shapes
:Гt
#loop_body/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      £
loop_body/Tensordot_2/ReshapeReshape"loop_body/ReadVariableOp_2:value:0,loop_body/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes

:y
$loop_body/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ©
loop_body/Tensordot_2/transpose	Transposeloop_body/Tensordot_1:output:0-loop_body/Tensordot_2/transpose/perm:output:0*
T0*#
_output_shapes
:Гv
%loop_body/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   Г   ©
loop_body/Tensordot_2/Reshape_1Reshape#loop_body/Tensordot_2/transpose:y:0.loop_body/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	ГҐ
loop_body/Tensordot_2/MatMulMatMul&loop_body/Tensordot_2/Reshape:output:0(loop_body/Tensordot_2/Reshape_1:output:0*
T0*
_output_shapes
:	Гf
loop_body/Tensordot_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:ГФ
loop_body/Tensordot_2Reshape&loop_body/Tensordot_2/MatMul:product:0$loop_body/Tensordot_2/shape:output:0*
T0*
_output_shapes	
:Г
loop_body/add/ReadVariableOpReadVariableOp%loop_body_add_readvariableop_resource*
_output_shapes	
:Г*
dtype0В
loop_body/addAddV2loop_body/Tensordot_2:output:0$loop_body/add/ReadVariableOp:value:0*
T0*
_output_shapes	
:Г\
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
:€€€€€€€€€^
loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : _
loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ф
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
value	B :Ъ
loop_body/SelectV2/pfor/add_1AddV2'loop_body/SelectV2/pfor/Rank_2:output:0(loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: Х
loop_body/SelectV2/pfor/MaximumMaximum'loop_body/SelectV2/pfor/Rank_1:output:0loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: Х
!loop_body/SelectV2/pfor/Maximum_1Maximum!loop_body/SelectV2/pfor/add_1:z:0#loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: `
loop_body/SelectV2/pfor/ShapeShapepfor/range:output:0*
T0*
_output_shapes
:У
loop_body/SelectV2/pfor/subSub%loop_body/SelectV2/pfor/Maximum_1:z:0'loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: o
%loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:†
loop_body/SelectV2/pfor/ReshapeReshapeloop_body/SelectV2/pfor/sub:z:0.loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:l
"loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Ю
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
valueB:«
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
valueB:Ћ
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
value	B : П
loop_body/SelectV2/pfor/concatConcatV2.loop_body/SelectV2/pfor/strided_slice:output:0%loop_body/SelectV2/pfor/Tile:output:00loop_body/SelectV2/pfor/strided_slice_1:output:0,loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ш
!loop_body/SelectV2/pfor/Reshape_1Reshapepfor/range:output:0'loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:€€€€€€€€€Љ
 loop_body/SelectV2/pfor/SelectV2SelectV2loop_body/Greater:z:0*loop_body/SelectV2/pfor/Reshape_1:output:0loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:€€€€€€€€€g
%loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : д
 loop_body/GatherV2/pfor/GatherV2GatherV2inputs)loop_body/SelectV2/pfor/SelectV2:output:0.loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:€€€€€€€€€@d
"loop_body/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : љ
loop_body/Reshape/pfor/concatConcatV2pfor/Reshape:output:0 loop_body/Reshape/shape:output:0+loop_body/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ґ
loop_body/Reshape/pfor/ReshapeReshape)loop_body/GatherV2/pfor/GatherV2:output:0&loop_body/Reshape/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€j
(loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :і
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
value	B : Е
)loop_body/Tensordot/transpose/pfor/concatConcatV2;loop_body/Tensordot/transpose/pfor/concat/values_0:output:0*loop_body/Tensordot/transpose/pfor/add:z:07loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:–
,loop_body/Tensordot/transpose/pfor/Transpose	Transpose'loop_body/Reshape/pfor/Reshape:output:02loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€n
,loop_body/Tensordot/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : џ
'loop_body/Tensordot/Reshape/pfor/concatConcatV2pfor/Reshape:output:0*loop_body/Tensordot/Reshape/shape:output:05loop_body/Tensordot/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ќ
(loop_body/Tensordot/Reshape/pfor/ReshapeReshape0loop_body/Tensordot/transpose/pfor/Transpose:y:00loop_body/Tensordot/Reshape/pfor/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€Ж
%loop_body/Tensordot/MatMul/pfor/ShapeShape1loop_body/Tensordot/Reshape/pfor/Reshape:output:0*
T0*
_output_shapes
:q
/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Џ
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
valueB љ
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
valueB Ѕ
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
valueB Ѕ
)loop_body/Tensordot/MatMul/pfor/Reshape_2Reshape.loop_body/Tensordot/MatMul/pfor/split:output:2:loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ±
#loop_body/Tensordot/MatMul/pfor/mulMul0loop_body/Tensordot/MatMul/pfor/Reshape:output:02loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: ¬
/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack'loop_body/Tensordot/MatMul/pfor/mul:z:02loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:№
)loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape1loop_body/Tensordot/Reshape/pfor/Reshape:output:08loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€Є
&loop_body/Tensordot/MatMul/pfor/MatMulMatMul2loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0 loop_body/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€|
1loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€З
/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePack0loop_body/Tensordot/MatMul/pfor/Reshape:output:02loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0:loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:÷
)loop_body/Tensordot/MatMul/pfor/Reshape_4Reshape0loop_body/Tensordot/MatMul/pfor/MatMul:product:08loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€f
$loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : √
loop_body/Tensordot/pfor/concatConcatV2pfor/Reshape:output:0"loop_body/Tensordot/shape:output:0-loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:√
 loop_body/Tensordot/pfor/ReshapeReshape2loop_body/Tensordot/MatMul/pfor/Reshape_4:output:0(loop_body/Tensordot/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€n
,loop_body/Tensordot_1/transpose_1/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :ј
*loop_body/Tensordot_1/transpose_1/pfor/addAddV2/loop_body/Tensordot_1/transpose_1/perm:output:05loop_body/Tensordot_1/transpose_1/pfor/add/y:output:0*
T0*
_output_shapes
:А
6loop_body/Tensordot_1/transpose_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: t
2loop_body/Tensordot_1/transpose_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Х
-loop_body/Tensordot_1/transpose_1/pfor/concatConcatV2?loop_body/Tensordot_1/transpose_1/pfor/concat/values_0:output:0.loop_body/Tensordot_1/transpose_1/pfor/add:z:0;loop_body/Tensordot_1/transpose_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Џ
0loop_body/Tensordot_1/transpose_1/pfor/Transpose	Transpose)loop_body/Tensordot/pfor/Reshape:output:06loop_body/Tensordot_1/transpose_1/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€r
0loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : з
+loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2pfor/Reshape:output:0.loop_body/Tensordot_1/Reshape_1/shape:output:09loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ў
,loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape4loop_body/Tensordot_1/transpose_1/pfor/Transpose:y:04loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€М
'loop_body/Tensordot_1/MatMul/pfor/ShapeShape5loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
5loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0>loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЋ
)loop_body/Tensordot_1/MatMul/pfor/MinimumMinimum8loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: k
)loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :і
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
valueB"          о
(loop_body/Tensordot_1/MatMul/pfor/SelectSelect+loop_body/Tensordot_1/MatMul/pfor/Equal:z:03loop_body/Tensordot_1/MatMul/pfor/Select/t:output:03loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:Б
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskС
'loop_body/Tensordot_1/MatMul/pfor/stackPack:loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:к
+loop_body/Tensordot_1/MatMul/pfor/transpose	Transpose5loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:01loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€Ќ
)loop_body/Tensordot_1/MatMul/pfor/ReshapeReshape/loop_body/Tensordot_1/MatMul/pfor/transpose:y:00loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*+
_output_shapes
:€€€€€€€€€Л
)loop_body/Tensordot_1/MatMul/pfor/Shape_1Shape2loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:s
1loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : в
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
valueB «
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
valueB «
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
valueB «
+loop_body/Tensordot_1/MatMul/pfor/Reshape_3Reshape0loop_body/Tensordot_1/MatMul/pfor/split:output:2<loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: є
%loop_body/Tensordot_1/MatMul/pfor/mulMul4loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: »
1loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePack4loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:0)loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:б
+loop_body/Tensordot_1/MatMul/pfor/Reshape_4Reshape2loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0:loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€√
(loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul&loop_body/Tensordot_1/Reshape:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*(
_output_shapes
:М€€€€€€€€€~
3loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€С
1loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePack<loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:о
+loop_body/Tensordot_1/MatMul/pfor/Reshape_5Reshape2loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0:loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€З
2loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          д
-loop_body/Tensordot_1/MatMul/pfor/transpose_1	Transpose4loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0;loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Мh
&loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : …
!loop_body/Tensordot_1/pfor/concatConcatV2pfor/Reshape:output:0$loop_body/Tensordot_1/shape:output:0/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:«
"loop_body/Tensordot_1/pfor/ReshapeReshape1loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:0*loop_body/Tensordot_1/pfor/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€Гl
*loop_body/Tensordot_2/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ї
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
value	B : Н
+loop_body/Tensordot_2/transpose/pfor/concatConcatV2=loop_body/Tensordot_2/transpose/pfor/concat/values_0:output:0,loop_body/Tensordot_2/transpose/pfor/add:z:09loop_body/Tensordot_2/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ў
.loop_body/Tensordot_2/transpose/pfor/Transpose	Transpose+loop_body/Tensordot_1/pfor/Reshape:output:04loop_body/Tensordot_2/transpose/pfor/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€Гr
0loop_body/Tensordot_2/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : з
+loop_body/Tensordot_2/Reshape_1/pfor/concatConcatV2pfor/Reshape:output:0.loop_body/Tensordot_2/Reshape_1/shape:output:09loop_body/Tensordot_2/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ў
,loop_body/Tensordot_2/Reshape_1/pfor/ReshapeReshape2loop_body/Tensordot_2/transpose/pfor/Transpose:y:04loop_body/Tensordot_2/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€ГМ
'loop_body/Tensordot_2/MatMul/pfor/ShapeShape5loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
5loop_body/Tensordot_2/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
/loop_body/Tensordot_2/MatMul/pfor/strided_sliceStridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0>loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_1StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЋ
)loop_body/Tensordot_2/MatMul/pfor/MinimumMinimum8loop_body/Tensordot_2/MatMul/pfor/strided_slice:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: k
)loop_body/Tensordot_2/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :і
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
valueB"          о
(loop_body/Tensordot_2/MatMul/pfor/SelectSelect+loop_body/Tensordot_2/MatMul/pfor/Equal:z:03loop_body/Tensordot_2/MatMul/pfor/Select/t:output:03loop_body/Tensordot_2/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:Б
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_2StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_3StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_4StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskС
'loop_body/Tensordot_2/MatMul/pfor/stackPack:loop_body/Tensordot_2/MatMul/pfor/strided_slice_2:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_3:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:к
+loop_body/Tensordot_2/MatMul/pfor/transpose	Transpose5loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:01loop_body/Tensordot_2/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€ќ
)loop_body/Tensordot_2/MatMul/pfor/ReshapeReshape/loop_body/Tensordot_2/MatMul/pfor/transpose:y:00loop_body/Tensordot_2/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:€€€€€€€€€ГЛ
)loop_body/Tensordot_2/MatMul/pfor/Shape_1Shape2loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:s
1loop_body/Tensordot_2/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : в
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
valueB «
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
valueB «
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
valueB «
+loop_body/Tensordot_2/MatMul/pfor/Reshape_3Reshape0loop_body/Tensordot_2/MatMul/pfor/split:output:2<loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: є
%loop_body/Tensordot_2/MatMul/pfor/mulMul4loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: »
1loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shapePack4loop_body/Tensordot_2/MatMul/pfor/Reshape_1:output:0)loop_body/Tensordot_2/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:б
+loop_body/Tensordot_2/MatMul/pfor/Reshape_4Reshape2loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0:loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€¬
(loop_body/Tensordot_2/MatMul/pfor/MatMulMatMul&loop_body/Tensordot_2/Reshape:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€~
3loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€С
1loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shapePack<loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:о
+loop_body/Tensordot_2/MatMul/pfor/Reshape_5Reshape2loop_body/Tensordot_2/MatMul/pfor/MatMul:product:0:loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€З
2loop_body/Tensordot_2/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          д
-loop_body/Tensordot_2/MatMul/pfor/transpose_1	Transpose4loop_body/Tensordot_2/MatMul/pfor/Reshape_5:output:0;loop_body/Tensordot_2/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Гh
&loop_body/Tensordot_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : …
!loop_body/Tensordot_2/pfor/concatConcatV2pfor/Reshape:output:0$loop_body/Tensordot_2/shape:output:0/loop_body/Tensordot_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:њ
"loop_body/Tensordot_2/pfor/ReshapeReshape1loop_body/Tensordot_2/MatMul/pfor/transpose_1:y:0*loop_body/Tensordot_2/pfor/concat:output:0*
T0*(
_output_shapes
:€€€€€€€€€ГY
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
value	B :З
loop_body/add/pfor/addAddV2"loop_body/add/pfor/Rank_1:output:0!loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: Д
loop_body/add/pfor/MaximumMaximumloop_body/add/pfor/add:z:0 loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: s
loop_body/add/pfor/ShapeShape+loop_body/Tensordot_2/pfor/Reshape:output:0*
T0*
_output_shapes
:А
loop_body/add/pfor/subSubloop_body/add/pfor/Maximum:z:0 loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: j
 loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:С
loop_body/add/pfor/ReshapeReshapeloop_body/add/pfor/sub:z:0)loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:g
loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:П
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
valueB:Ѓ
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
valueB:і
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
value	B : ц
loop_body/add/pfor/concatConcatV2)loop_body/add/pfor/strided_slice:output:0 loop_body/add/pfor/Tile:output:0+loop_body/add/pfor/strided_slice_1:output:0'loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ђ
loop_body/add/pfor/Reshape_1Reshape+loop_body/Tensordot_2/pfor/Reshape:output:0"loop_body/add/pfor/concat:output:0*
T0*(
_output_shapes
:€€€€€€€€€Г°
loop_body/add/pfor/AddV2AddV2%loop_body/add/pfor/Reshape_1:output:0$loop_body/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Г^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Г   {
ReshapeReshapeloop_body/add/pfor/AddV2:z:0Reshape/shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ГW
SoftmaxSoftmaxReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€Гa
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ГЇ
NoOpNoOp^loop_body/ReadVariableOp^loop_body/ReadVariableOp_1^loop_body/ReadVariableOp_2^loop_body/add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 24
loop_body/ReadVariableOploop_body/ReadVariableOp28
loop_body/ReadVariableOp_1loop_body/ReadVariableOp_128
loop_body/ReadVariableOp_2loop_body/ReadVariableOp_22<
loop_body/add/ReadVariableOploop_body/add/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Е
љ
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
	unknown_8:	А
	unknown_9:%

unknown_10:А 

unknown_11: 

unknown_12:

unknown_13:%

unknown_14:Г

unknown_15:

unknown_16:	Г
identityИҐStatefulPartitionedCallƒ
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
:€€€€€€€€€Г: : : : *4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_413197p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Г`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€22: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs
„
b
D__inference_dropout1_layer_call_and_return_conditional_losses_412879

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
У
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_412324

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
≥9
Љ
E__inference_conv2dmpo_layer_call_and_return_conditional_losses_415796

inputs1
readvariableop_resource:3
readvariableop_1_resource:/
!reshape_2_readvariableop_resource:
identityИҐReadVariableOpҐReadVariableOp_1ҐReshape_2/ReadVariableOpn
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
valueB"             О
Tensordot/transpose	TransposeReadVariableOp_1:value:0!Tensordot/transpose/perm:output:0*
T0*&
_output_shapes
:h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      А
Tensordot/ReshapeReshapeTensordot/transpose:y:0 Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:s
Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             Р
Tensordot/transpose_1	TransposeReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*&
_output_shapes
:j
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ж
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
value$B""                  Г
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
valueB:Ќ
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
valueB:„
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
€€€€€€€€€К
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
valueB:„
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
valueB:ў
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
€€€€€€€€€Р
concat_1ConcatV2strided_slice_3:output:0concat_1/values_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:i
	Reshape_1Reshapetranspose_2:y:0concat_1:output:0*
T0*&
_output_shapes
:О
Conv2DConv2DinputsReshape_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€//*
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
:€€€€€€€€€//O
ReluReluadd:z:0*
T0*/
_output_shapes
:€€€€€€€€€//i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€//Е
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^Reshape_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€//: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_124
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€//
 
_user_specified_nameinputs
С
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_412299

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ѕ^
∞
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
conv2dmpo_1_413644:	А&
fruit_tn1_413656:+
fruit_tn1_413658:А&
fruit_tn1_413660:&
fruit_tn1_413662:"
fruit_tn2_413674:+
fruit_tn2_413676:Г"
fruit_tn2_413678:
fruit_tn2_413680:	Г
identity

identity_1

identity_2

identity_3

identity_4ИҐconv2d/StatefulPartitionedCallҐ!conv2dmpo/StatefulPartitionedCallҐ#conv2dmpo_1/StatefulPartitionedCallҐ!fruit_tn1/StatefulPartitionedCallҐ!fruit_tn2/StatefulPartitionedCallх
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_12conv2d_413615conv2d_413617*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€//*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_412370і
!conv2dmpo/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2dmpo_413620conv2dmpo_413622conv2dmpo_413624*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€//*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2dmpo_layer_call_and_return_conditional_losses_412438–
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
GPU2*0J 8В *:
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
valueB:з
+conv2dmpo/ActivityRegularizer/strided_sliceStridedSlice,conv2dmpo/ActivityRegularizer/Shape:output:0:conv2dmpo/ActivityRegularizer/strided_slice/stack:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_1:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
"conv2dmpo/ActivityRegularizer/CastCast4conv2dmpo/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ±
%conv2dmpo/ActivityRegularizer/truedivRealDiv6conv2dmpo/ActivityRegularizer/PartitionedCall:output:0&conv2dmpo/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: с
max_pooling2d/PartitionedCallPartitionedCall*conv2dmpo/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_412299к
#conv2dmpo_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2dmpo_1_413636conv2dmpo_1_413638conv2dmpo_1_413640conv2dmpo_1_413642conv2dmpo_1_413644*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2dmpo_1_layer_call_and_return_conditional_losses_412543÷
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
GPU2*0J 8В *<
f7R5
3__inference_conv2dmpo_1_activity_regularizer_412315Б
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
valueB:с
-conv2dmpo_1/ActivityRegularizer/strided_sliceStridedSlice.conv2dmpo_1/ActivityRegularizer/Shape:output:0<conv2dmpo_1/ActivityRegularizer/strided_slice/stack:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_1:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskФ
$conv2dmpo_1/ActivityRegularizer/CastCast6conv2dmpo_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ј
'conv2dmpo_1/ActivityRegularizer/truedivRealDiv8conv2dmpo_1/ActivityRegularizer/PartitionedCall:output:0(conv2dmpo_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ш
max_pooling2d_1/PartitionedCallPartitionedCall,conv2dmpo_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_412324Ѕ
!fruit_tn1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0fruit_tn1_413656fruit_tn1_413658fruit_tn1_413660fruit_tn1_413662*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_fruit_tn1_layer_call_and_return_conditional_losses_412856–
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
GPU2*0J 8В *:
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
valueB:з
+fruit_tn1/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn1/ActivityRegularizer/Shape:output:0:fruit_tn1/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
"fruit_tn1/ActivityRegularizer/CastCast4fruit_tn1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ±
%fruit_tn1/ActivityRegularizer/truedivRealDiv6fruit_tn1/ActivityRegularizer/PartitionedCall:output:0&fruit_tn1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: я
dropout1/PartitionedCallPartitionedCall*fruit_tn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout1_layer_call_and_return_conditional_losses_412879ї
!fruit_tn2/StatefulPartitionedCallStatefulPartitionedCall!dropout1/PartitionedCall:output:0fruit_tn2_413674fruit_tn2_413676fruit_tn2_413678fruit_tn2_413680*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Г*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_fruit_tn2_layer_call_and_return_conditional_losses_413174–
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
GPU2*0J 8В *:
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
valueB:з
+fruit_tn2/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn2/ActivityRegularizer/Shape:output:0:fruit_tn2/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
"fruit_tn2/ActivityRegularizer/CastCast4fruit_tn2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ±
%fruit_tn2/ActivityRegularizer/truedivRealDiv6fruit_tn2/ActivityRegularizer/PartitionedCall:output:0&fruit_tn2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: z
IdentityIdentity*fruit_tn2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Гi

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
: щ
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
?:€€€€€€€€€22: : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2F
!conv2dmpo/StatefulPartitionedCall!conv2dmpo/StatefulPartitionedCall2J
#conv2dmpo_1/StatefulPartitionedCall#conv2dmpo_1/StatefulPartitionedCall2F
!fruit_tn1/StatefulPartitionedCall!fruit_tn1/StatefulPartitionedCall2F
!fruit_tn2/StatefulPartitionedCall!fruit_tn2/StatefulPartitionedCall:Y U
/
_output_shapes
:€€€€€€€€€22
"
_user_specified_name
input_12
л
ј
K__inference_conv2dmpo_1_layer_call_and_return_all_conditional_losses_415625

inputs!
unknown:#
	unknown_0:#
	unknown_1:#
	unknown_2:
	unknown_3:	А
identity

identity_1ИҐStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2dmpo_1_layer_call_and_return_conditional_losses_412543™
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
GPU2*0J 8В *<
f7R5
3__inference_conv2dmpo_1_activity_regularizer_412315x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€АX

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
%:€€€€€€€€€: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
°
Щ
I__inference_fruit_tn1_layer_call_and_return_all_conditional_losses_415671

inputs
unknown:$
	unknown_0:А
	unknown_1:
	unknown_2:
identity

identity_1ИҐStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_fruit_tn1_layer_call_and_return_conditional_losses_412856®
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
GPU2*0J 8В *:
f5R3
1__inference_fruit_tn1_activity_regularizer_412340o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@X

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
$:€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ж√
щ3
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
#assignvariableop_9_conv2dmpo_1_bias:	А+
assignvariableop_10_a:0
assignvariableop_11_b:А+
assignvariableop_12_c:.
assignvariableop_13_bias:)
assignvariableop_14_a_1:2
assignvariableop_15_b_1:Г)
assignvariableop_16_c_1:)
assignvariableop_17_bias_1:	Г'
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
+assignvariableop_35_adam_conv2dmpo_1_bias_m:	А2
assignvariableop_36_adam_a_m:7
assignvariableop_37_adam_b_m:А2
assignvariableop_38_adam_c_m:5
assignvariableop_39_adam_bias_m:0
assignvariableop_40_adam_a_m_1:9
assignvariableop_41_adam_b_m_1:Г0
assignvariableop_42_adam_c_m_1:0
!assignvariableop_43_adam_bias_m_1:	ГB
(assignvariableop_44_adam_conv2d_kernel_v:4
&assignvariableop_45_adam_conv2d_bias_v:M
3assignvariableop_46_adam_conv2dmpo_end_node_first_v:L
2assignvariableop_47_adam_conv2dmpo_end_node_last_v:7
)assignvariableop_48_adam_conv2dmpo_bias_v:O
5assignvariableop_49_adam_conv2dmpo_1_end_node_first_v:N
4assignvariableop_50_adam_conv2dmpo_1_middle_node_0_v:N
4assignvariableop_51_adam_conv2dmpo_1_middle_node_1_v:N
4assignvariableop_52_adam_conv2dmpo_1_end_node_last_v::
+assignvariableop_53_adam_conv2dmpo_1_bias_v:	А2
assignvariableop_54_adam_a_v:7
assignvariableop_55_adam_b_v:А2
assignvariableop_56_adam_c_v:5
assignvariableop_57_adam_bias_v:0
assignvariableop_58_adam_a_v_1:9
assignvariableop_59_adam_b_v_1:Г0
assignvariableop_60_adam_c_v_1:0
!assignvariableop_61_adam_bias_v_1:	ГE
+assignvariableop_62_adam_conv2d_kernel_vhat:7
)assignvariableop_63_adam_conv2d_bias_vhat:P
6assignvariableop_64_adam_conv2dmpo_end_node_first_vhat:O
5assignvariableop_65_adam_conv2dmpo_end_node_last_vhat::
,assignvariableop_66_adam_conv2dmpo_bias_vhat:R
8assignvariableop_67_adam_conv2dmpo_1_end_node_first_vhat:Q
7assignvariableop_68_adam_conv2dmpo_1_middle_node_0_vhat:Q
7assignvariableop_69_adam_conv2dmpo_1_middle_node_1_vhat:Q
7assignvariableop_70_adam_conv2dmpo_1_end_node_last_vhat:=
.assignvariableop_71_adam_conv2dmpo_1_bias_vhat:	А5
assignvariableop_72_adam_a_vhat::
assignvariableop_73_adam_b_vhat:А5
assignvariableop_74_adam_c_vhat:8
"assignvariableop_75_adam_bias_vhat:3
!assignvariableop_76_adam_a_vhat_1:<
!assignvariableop_77_adam_b_vhat_1:Г3
!assignvariableop_78_adam_c_vhat_1:3
$assignvariableop_79_adam_bias_vhat_1:	Г
identity_81ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_67ҐAssignVariableOp_68ҐAssignVariableOp_69ҐAssignVariableOp_7ҐAssignVariableOp_70ҐAssignVariableOp_71ҐAssignVariableOp_72ҐAssignVariableOp_73ҐAssignVariableOp_74ҐAssignVariableOp_75ҐAssignVariableOp_76ҐAssignVariableOp_77ҐAssignVariableOp_78ҐAssignVariableOp_79ҐAssignVariableOp_8ҐAssignVariableOp_9ч0
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*Э0
valueУ0BР0QB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/end_node_first/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-1/end_node_last/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/end_node_first/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-2/middle_node_0/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-2/middle_node_1/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-2/end_node_last/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/a_var/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/b_var/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/c_var/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/a_var/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/b_var/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/c_var/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/end_node_first/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/end_node_last/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/end_node_first/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/middle_node_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/middle_node_1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/end_node_last/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/a_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/b_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/c_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/a_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/b_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/c_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/end_node_first/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/end_node_last/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/end_node_first/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/middle_node_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/middle_node_1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/end_node_last/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/a_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/b_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/c_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/a_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/b_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/c_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-1/end_node_first/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/end_node_last/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-2/end_node_first/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/middle_node_0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/middle_node_1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/end_node_last/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/a_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/b_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/c_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/a_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/b_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/c_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHХ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*Ј
value≠B™QB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ґ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Џ
_output_shapes«
ƒ:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*_
dtypesU
S2Q	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_2AssignVariableOp+assignvariableop_2_conv2dmpo_end_node_firstIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_3AssignVariableOp*assignvariableop_3_conv2dmpo_end_node_lastIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_4AssignVariableOp!assignvariableop_4_conv2dmpo_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_5AssignVariableOp-assignvariableop_5_conv2dmpo_1_end_node_firstIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_6AssignVariableOp,assignvariableop_6_conv2dmpo_1_middle_node_0Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_7AssignVariableOp,assignvariableop_7_conv2dmpo_1_middle_node_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_8AssignVariableOp,assignvariableop_8_conv2dmpo_1_end_node_lastIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_9AssignVariableOp#assignvariableop_9_conv2dmpo_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_10AssignVariableOpassignvariableop_10_aIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_11AssignVariableOpassignvariableop_11_bIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_12AssignVariableOpassignvariableop_12_cIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_13AssignVariableOpassignvariableop_13_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_14AssignVariableOpassignvariableop_14_a_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_15AssignVariableOpassignvariableop_15_b_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_16AssignVariableOpassignvariableop_16_c_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_17AssignVariableOpassignvariableop_17_bias_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:О
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_beta_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_decayIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_24AssignVariableOp"assignvariableop_24_true_positivesIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_25AssignVariableOp#assignvariableop_25_false_positivesIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_27AssignVariableOp&assignvariableop_27_adam_conv2d_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_28AssignVariableOp3assignvariableop_28_adam_conv2dmpo_end_node_first_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_29AssignVariableOp2assignvariableop_29_adam_conv2dmpo_end_node_last_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2dmpo_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_31AssignVariableOp5assignvariableop_31_adam_conv2dmpo_1_end_node_first_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_32AssignVariableOp4assignvariableop_32_adam_conv2dmpo_1_middle_node_0_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_33AssignVariableOp4assignvariableop_33_adam_conv2dmpo_1_middle_node_1_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adam_conv2dmpo_1_end_node_last_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2dmpo_1_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_36AssignVariableOpassignvariableop_36_adam_a_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_37AssignVariableOpassignvariableop_37_adam_b_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_38AssignVariableOpassignvariableop_38_adam_c_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_39AssignVariableOpassignvariableop_39_adam_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_40AssignVariableOpassignvariableop_40_adam_a_m_1Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_41AssignVariableOpassignvariableop_41_adam_b_m_1Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_42AssignVariableOpassignvariableop_42_adam_c_m_1Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_43AssignVariableOp!assignvariableop_43_adam_bias_m_1Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv2d_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_45AssignVariableOp&assignvariableop_45_adam_conv2d_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_46AssignVariableOp3assignvariableop_46_adam_conv2dmpo_end_node_first_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_47AssignVariableOp2assignvariableop_47_adam_conv2dmpo_end_node_last_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2dmpo_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_49AssignVariableOp5assignvariableop_49_adam_conv2dmpo_1_end_node_first_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_50AssignVariableOp4assignvariableop_50_adam_conv2dmpo_1_middle_node_0_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_51AssignVariableOp4assignvariableop_51_adam_conv2dmpo_1_middle_node_1_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_52AssignVariableOp4assignvariableop_52_adam_conv2dmpo_1_end_node_last_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_conv2dmpo_1_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_54AssignVariableOpassignvariableop_54_adam_a_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_55AssignVariableOpassignvariableop_55_adam_b_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_56AssignVariableOpassignvariableop_56_adam_c_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_57AssignVariableOpassignvariableop_57_adam_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_58AssignVariableOpassignvariableop_58_adam_a_v_1Identity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_59AssignVariableOpassignvariableop_59_adam_b_v_1Identity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_60AssignVariableOpassignvariableop_60_adam_c_v_1Identity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_61AssignVariableOp!assignvariableop_61_adam_bias_v_1Identity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_62AssignVariableOp+assignvariableop_62_adam_conv2d_kernel_vhatIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_63AssignVariableOp)assignvariableop_63_adam_conv2d_bias_vhatIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_conv2dmpo_end_node_first_vhatIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_65AssignVariableOp5assignvariableop_65_adam_conv2dmpo_end_node_last_vhatIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_66AssignVariableOp,assignvariableop_66_adam_conv2dmpo_bias_vhatIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_67AssignVariableOp8assignvariableop_67_adam_conv2dmpo_1_end_node_first_vhatIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_68AssignVariableOp7assignvariableop_68_adam_conv2dmpo_1_middle_node_0_vhatIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_69AssignVariableOp7assignvariableop_69_adam_conv2dmpo_1_middle_node_1_vhatIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_70AssignVariableOp7assignvariableop_70_adam_conv2dmpo_1_end_node_last_vhatIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_71AssignVariableOp.assignvariableop_71_adam_conv2dmpo_1_bias_vhatIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_72AssignVariableOpassignvariableop_72_adam_a_vhatIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_73AssignVariableOpassignvariableop_73_adam_b_vhatIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_74AssignVariableOpassignvariableop_74_adam_c_vhatIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_75AssignVariableOp"assignvariableop_75_adam_bias_vhatIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_76AssignVariableOp!assignvariableop_76_adam_a_vhat_1Identity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_77AssignVariableOp!assignvariableop_77_adam_b_vhat_1Identity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_78AssignVariableOp!assignvariableop_78_adam_c_vhat_1Identity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_79AssignVariableOp$assignvariableop_79_adam_bias_vhat_1Identity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ѓ
Identity_80Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_81IdentityIdentity_80:output:0^NoOp_1*
T0*
_output_shapes
: Ь
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_81Identity_81:output:0*Ј
_input_shapes•
Ґ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
¶

ы
B__inference_conv2d_layer_call_and_return_conditional_losses_415559

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€//*
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
:€€€€€€€€€//g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€//w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs
ъ
у
I__inference_conv2dmpo_layer_call_and_return_all_conditional_losses_415583

inputs!
unknown:#
	unknown_0:
	unknown_1:
identity

identity_1ИҐStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€//*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2dmpo_layer_call_and_return_conditional_losses_412438®
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
GPU2*0J 8В *:
f5R3
1__inference_conv2dmpo_activity_regularizer_412290w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€//X

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
!:€€€€€€€€€//: : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€//
 
_user_specified_nameinputs
т
b
)__inference_dropout1_layer_call_fn_415681

inputs
identityИҐStatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout1_layer_call_and_return_conditional_losses_413290o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
†
E
)__inference_dropout1_layer_call_fn_415676

inputs
identity≤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout1_layer_call_and_return_conditional_losses_412879`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
щ°
≠
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
-conv2dmpo_1_reshape_2_readvariableop_resource:	АA
+fruit_tn1_loop_body_readvariableop_resource:H
-fruit_tn1_loop_body_readvariableop_1_resource:АC
-fruit_tn1_loop_body_readvariableop_2_resource:E
/fruit_tn1_loop_body_add_readvariableop_resource:=
+fruit_tn2_loop_body_readvariableop_resource:H
-fruit_tn2_loop_body_readvariableop_1_resource:Г?
-fruit_tn2_loop_body_readvariableop_2_resource:>
/fruit_tn2_loop_body_add_readvariableop_resource:	Г
identity

identity_1

identity_2

identity_3

identity_4ИҐconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOpҐconv2dmpo/ReadVariableOpҐconv2dmpo/ReadVariableOp_1Ґ"conv2dmpo/Reshape_2/ReadVariableOpҐconv2dmpo_1/ReadVariableOpҐconv2dmpo_1/ReadVariableOp_1Ґconv2dmpo_1/ReadVariableOp_2Ґconv2dmpo_1/ReadVariableOp_3Ґ$conv2dmpo_1/Reshape_2/ReadVariableOpҐ"fruit_tn1/loop_body/ReadVariableOpҐ$fruit_tn1/loop_body/ReadVariableOp_1Ґ$fruit_tn1/loop_body/ReadVariableOp_2Ґ&fruit_tn1/loop_body/add/ReadVariableOpҐ"fruit_tn2/loop_body/ReadVariableOpҐ$fruit_tn2/loop_body/ReadVariableOp_1Ґ$fruit_tn2/loop_body/ReadVariableOp_2Ґ&fruit_tn2/loop_body/add/ReadVariableOpК
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0®
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€//*
paddingVALID*
strides
А
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Т
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€//В
conv2dmpo/ReadVariableOpReadVariableOp!conv2dmpo_readvariableop_resource*&
_output_shapes
:*
dtype0Ж
conv2dmpo/ReadVariableOp_1ReadVariableOp#conv2dmpo_readvariableop_1_resource*&
_output_shapes
:*
dtype0{
"conv2dmpo/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ™
conv2dmpo/Tensordot/transpose	Transpose conv2dmpo/ReadVariableOp:value:0+conv2dmpo/Tensordot/transpose/perm:output:0*
T0*&
_output_shapes
:r
!conv2dmpo/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ю
conv2dmpo/Tensordot/ReshapeReshape!conv2dmpo/Tensordot/transpose:y:0*conv2dmpo/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:}
$conv2dmpo/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ∞
conv2dmpo/Tensordot/transpose_1	Transpose"conv2dmpo/ReadVariableOp_1:value:0-conv2dmpo/Tensordot/transpose_1/perm:output:0*
T0*&
_output_shapes
:t
#conv2dmpo/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      §
conv2dmpo/Tensordot/Reshape_1Reshape#conv2dmpo/Tensordot/transpose_1:y:0,conv2dmpo/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:Ы
conv2dmpo/Tensordot/MatMulMatMul$conv2dmpo/Tensordot/Reshape:output:0&conv2dmpo/Tensordot/Reshape_1:output:0*
T0*
_output_shapes

:z
conv2dmpo/Tensordot/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  °
conv2dmpo/TensordotReshape$conv2dmpo/Tensordot/MatMul:product:0"conv2dmpo/Tensordot/shape:output:0*
T0*.
_output_shapes
:y
conv2dmpo/transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   Ъ
conv2dmpo/transpose	Transposeconv2dmpo/Tensordot:output:0!conv2dmpo/transpose/perm:output:0*
T0*.
_output_shapes
:{
conv2dmpo/transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   Щ
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
valueB:€
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
valueB:Й
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
€€€€€€€€€≤
conv2dmpo/concatConcatV2"conv2dmpo/strided_slice_1:output:0"conv2dmpo/concat/values_1:output:0conv2dmpo/concat/axis:output:0*
N*
T0*
_output_shapes
:З
conv2dmpo/ReshapeReshapeconv2dmpo/transpose_1:y:0conv2dmpo/concat:output:0*
T0**
_output_shapes
:w
conv2dmpo/transpose_2/permConst*
_output_shapes
:*
dtype0*)
value B"                Ш
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
valueB:Й
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
valueB:Л
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
€€€€€€€€€Є
conv2dmpo/concat_1ConcatV2"conv2dmpo/strided_slice_3:output:0$conv2dmpo/concat_1/values_1:output:0 conv2dmpo/concat_1/axis:output:0*
N*
T0*
_output_shapes
:З
conv2dmpo/Reshape_1Reshapeconv2dmpo/transpose_2:y:0conv2dmpo/concat_1:output:0*
T0*&
_output_shapes
:≥
conv2dmpo/Conv2DConv2Dconv2d/BiasAdd:output:0conv2dmpo/Reshape_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€//*
paddingSAME*
strides
К
"conv2dmpo/Reshape_2/ReadVariableOpReadVariableOp+conv2dmpo_reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0j
conv2dmpo/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ч
conv2dmpo/Reshape_2Reshape*conv2dmpo/Reshape_2/ReadVariableOp:value:0"conv2dmpo/Reshape_2/shape:output:0*
T0*
_output_shapes

:Й
conv2dmpo/addAddV2conv2dmpo/Conv2D:output:0conv2dmpo/Reshape_2:output:0*
T0*/
_output_shapes
:€€€€€€€€€//c
conv2dmpo/ReluReluconv2dmpo/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€//Ж
$conv2dmpo/ActivityRegularizer/SquareSquareconv2dmpo/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€//|
#conv2dmpo/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             °
!conv2dmpo/ActivityRegularizer/SumSum(conv2dmpo/ActivityRegularizer/Square:y:0,conv2dmpo/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: h
#conv2dmpo/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ђ≈'7£
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
valueB:з
+conv2dmpo/ActivityRegularizer/strided_sliceStridedSlice,conv2dmpo/ActivityRegularizer/Shape:output:0:conv2dmpo/ActivityRegularizer/strided_slice/stack:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_1:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
"conv2dmpo/ActivityRegularizer/CastCast4conv2dmpo/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: †
%conv2dmpo/ActivityRegularizer/truedivRealDiv%conv2dmpo/ActivityRegularizer/mul:z:0&conv2dmpo/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Ђ
max_pooling2d/MaxPoolMaxPoolconv2dmpo/Relu:activations:0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
Ж
conv2dmpo_1/ReadVariableOpReadVariableOp#conv2dmpo_1_readvariableop_resource*&
_output_shapes
:*
dtype0К
conv2dmpo_1/ReadVariableOp_1ReadVariableOp%conv2dmpo_1_readvariableop_1_resource*&
_output_shapes
:*
dtype0К
conv2dmpo_1/ReadVariableOp_2ReadVariableOp%conv2dmpo_1_readvariableop_2_resource*&
_output_shapes
:*
dtype0К
conv2dmpo_1/ReadVariableOp_3ReadVariableOp%conv2dmpo_1_readvariableop_3_resource*&
_output_shapes
:*
dtype0}
$conv2dmpo_1/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ≤
conv2dmpo_1/Tensordot/transpose	Transpose$conv2dmpo_1/ReadVariableOp_1:value:0-conv2dmpo_1/Tensordot/transpose/perm:output:0*
T0*&
_output_shapes
:t
#conv2dmpo_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@      §
conv2dmpo_1/Tensordot/ReshapeReshape#conv2dmpo_1/Tensordot/transpose:y:0,conv2dmpo_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@
&conv2dmpo_1/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             і
!conv2dmpo_1/Tensordot/transpose_1	Transpose"conv2dmpo_1/ReadVariableOp:value:0/conv2dmpo_1/Tensordot/transpose_1/perm:output:0*
T0*&
_output_shapes
:v
%conv2dmpo_1/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ™
conv2dmpo_1/Tensordot/Reshape_1Reshape%conv2dmpo_1/Tensordot/transpose_1:y:0.conv2dmpo_1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:°
conv2dmpo_1/Tensordot/MatMulMatMul&conv2dmpo_1/Tensordot/Reshape:output:0(conv2dmpo_1/Tensordot/Reshape_1:output:0*
T0*
_output_shapes

:@|
conv2dmpo_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  І
conv2dmpo_1/TensordotReshape&conv2dmpo_1/Tensordot/MatMul:product:0$conv2dmpo_1/Tensordot/shape:output:0*
T0*.
_output_shapes
:
&conv2dmpo_1/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ґ
!conv2dmpo_1/Tensordot_1/transpose	Transpose$conv2dmpo_1/ReadVariableOp_3:value:0/conv2dmpo_1/Tensordot_1/transpose/perm:output:0*
T0*&
_output_shapes
:v
%conv2dmpo_1/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ™
conv2dmpo_1/Tensordot_1/ReshapeReshape%conv2dmpo_1/Tensordot_1/transpose:y:0.conv2dmpo_1/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:Б
(conv2dmpo_1/Tensordot_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ї
#conv2dmpo_1/Tensordot_1/transpose_1	Transpose$conv2dmpo_1/ReadVariableOp_2:value:01conv2dmpo_1/Tensordot_1/transpose_1/perm:output:0*
T0*&
_output_shapes
:x
'conv2dmpo_1/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   ∞
!conv2dmpo_1/Tensordot_1/Reshape_1Reshape'conv2dmpo_1/Tensordot_1/transpose_1:y:00conv2dmpo_1/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:@І
conv2dmpo_1/Tensordot_1/MatMulMatMul(conv2dmpo_1/Tensordot_1/Reshape:output:0*conv2dmpo_1/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes

:@~
conv2dmpo_1/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  ≠
conv2dmpo_1/Tensordot_1Reshape(conv2dmpo_1/Tensordot_1/MatMul:product:0&conv2dmpo_1/Tensordot_1/shape:output:0*
T0*.
_output_shapes
:З
&conv2dmpo_1/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   Є
!conv2dmpo_1/Tensordot_2/transpose	Transposeconv2dmpo_1/Tensordot:output:0/conv2dmpo_1/Tensordot_2/transpose/perm:output:0*
T0*.
_output_shapes
:v
%conv2dmpo_1/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"А      Ђ
conv2dmpo_1/Tensordot_2/ReshapeReshape%conv2dmpo_1/Tensordot_2/transpose:y:0.conv2dmpo_1/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes
:	АЙ
(conv2dmpo_1/Tensordot_2/transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   Њ
#conv2dmpo_1/Tensordot_2/transpose_1	Transpose conv2dmpo_1/Tensordot_1:output:01conv2dmpo_1/Tensordot_2/transpose_1/perm:output:0*
T0*.
_output_shapes
:x
'conv2dmpo_1/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   А   ±
!conv2dmpo_1/Tensordot_2/Reshape_1Reshape'conv2dmpo_1/Tensordot_2/transpose_1:y:00conv2dmpo_1/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А©
conv2dmpo_1/Tensordot_2/MatMulMatMul(conv2dmpo_1/Tensordot_2/Reshape:output:0*conv2dmpo_1/Tensordot_2/Reshape_1:output:0*
T0* 
_output_shapes
:
ААО
conv2dmpo_1/Tensordot_2/shapeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                              љ
conv2dmpo_1/Tensordot_2Reshape(conv2dmpo_1/Tensordot_2/MatMul:product:0&conv2dmpo_1/Tensordot_2/shape:output:0*
T0*>
_output_shapes,
*:(Л
conv2dmpo_1/transpose/permConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                         	      ≤
conv2dmpo_1/transpose	Transpose conv2dmpo_1/Tensordot_2:output:0#conv2dmpo_1/transpose/perm:output:0*
T0*>
_output_shapes,
*:(Н
conv2dmpo_1/transpose_1/permConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                	               ѓ
conv2dmpo_1/transpose_1	Transposeconv2dmpo_1/transpose:y:0%conv2dmpo_1/transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(В
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
valueB:Й
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
valueB:У
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
€€€€€€€€€Ї
conv2dmpo_1/concatConcatV2$conv2dmpo_1/strided_slice_1:output:0$conv2dmpo_1/concat/values_1:output:0 conv2dmpo_1/concat/axis:output:0*
N*
T0*
_output_shapes
:Х
conv2dmpo_1/ReshapeReshapeconv2dmpo_1/transpose_1:y:0conv2dmpo_1/concat:output:0*
T0*2
_output_shapes 
:Б
conv2dmpo_1/transpose_2/permConst*
_output_shapes
:*
dtype0*1
value(B&"                      ¶
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
valueB:У
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
valueB:Х
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
€€€€€€€€€ј
conv2dmpo_1/concat_1ConcatV2$conv2dmpo_1/strided_slice_3:output:0&conv2dmpo_1/concat_1/values_1:output:0"conv2dmpo_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:О
conv2dmpo_1/Reshape_1Reshapeconv2dmpo_1/transpose_2:y:0conv2dmpo_1/concat_1:output:0*
T0*'
_output_shapes
:Ањ
conv2dmpo_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0conv2dmpo_1/Reshape_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
П
$conv2dmpo_1/Reshape_2/ReadVariableOpReadVariableOp-conv2dmpo_1_reshape_2_readvariableop_resource*
_output_shapes	
:А*
dtype0l
conv2dmpo_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ю
conv2dmpo_1/Reshape_2Reshape,conv2dmpo_1/Reshape_2/ReadVariableOp:value:0$conv2dmpo_1/Reshape_2/shape:output:0*
T0*
_output_shapes
:	АР
conv2dmpo_1/addAddV2conv2dmpo_1/Conv2D:output:0conv2dmpo_1/Reshape_2:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аh
conv2dmpo_1/ReluReluconv2dmpo_1/add:z:0*
T0*0
_output_shapes
:€€€€€€€€€АЛ
&conv2dmpo_1/ActivityRegularizer/SquareSquareconv2dmpo_1/Relu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€А~
%conv2dmpo_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             І
#conv2dmpo_1/ActivityRegularizer/SumSum*conv2dmpo_1/ActivityRegularizer/Square:y:0.conv2dmpo_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: j
%conv2dmpo_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ђ≈'7©
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
valueB:с
-conv2dmpo_1/ActivityRegularizer/strided_sliceStridedSlice.conv2dmpo_1/ActivityRegularizer/Shape:output:0<conv2dmpo_1/ActivityRegularizer/strided_slice/stack:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_1:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskФ
$conv2dmpo_1/ActivityRegularizer/CastCast6conv2dmpo_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ¶
'conv2dmpo_1/ActivityRegularizer/truedivRealDiv'conv2dmpo_1/ActivityRegularizer/mul:z:0(conv2dmpo_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ∞
max_pooling2d_1/MaxPoolMaxPoolconv2dmpo_1/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
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
valueB:Г
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
value	B :Н
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
value	B : ±
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
valueB:µ
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
value	B :Ы
fruit_tn1/loop_body/GreaterGreater*fruit_tn1/loop_body/strided_slice:output:0&fruit_tn1/loop_body/Greater/y:output:0*
T0*
_output_shapes
: `
fruit_tn1/loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : »
fruit_tn1/loop_body/SelectV2SelectV2fruit_tn1/loop_body/Greater:z:03fruit_tn1/loop_body/PlaceholderWithDefault:output:0'fruit_tn1/loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: c
!fruit_tn1/loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : о
fruit_tn1/loop_body/GatherV2GatherV2 max_pooling2d_1/MaxPool:output:0%fruit_tn1/loop_body/SelectV2:output:0*fruit_tn1/loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:АТ
"fruit_tn1/loop_body/ReadVariableOpReadVariableOp+fruit_tn1_loop_body_readvariableop_resource*"
_output_shapes
:*
dtype0Ы
$fruit_tn1/loop_body/ReadVariableOp_1ReadVariableOp-fruit_tn1_loop_body_readvariableop_1_resource*'
_output_shapes
:А*
dtype0Ц
$fruit_tn1/loop_body/ReadVariableOp_2ReadVariableOp-fruit_tn1_loop_body_readvariableop_2_resource*"
_output_shapes
:*
dtype0Б
,fruit_tn1/loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ј
'fruit_tn1/loop_body/Tensordot/transpose	Transpose%fruit_tn1/loop_body/GatherV2:output:05fruit_tn1/loop_body/Tensordot/transpose/perm:output:0*
T0*#
_output_shapes
:А|
+fruit_tn1/loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      љ
%fruit_tn1/loop_body/Tensordot/ReshapeReshape+fruit_tn1/loop_body/Tensordot/transpose:y:04fruit_tn1/loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	АГ
.fruit_tn1/loop_body/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          »
)fruit_tn1/loop_body/Tensordot/transpose_1	Transpose*fruit_tn1/loop_body/ReadVariableOp:value:07fruit_tn1/loop_body/Tensordot/transpose_1/perm:output:0*
T0*"
_output_shapes
:~
-fruit_tn1/loop_body/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ¬
'fruit_tn1/loop_body/Tensordot/Reshape_1Reshape-fruit_tn1/loop_body/Tensordot/transpose_1:y:06fruit_tn1/loop_body/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:Ї
$fruit_tn1/loop_body/Tensordot/MatMulMatMul.fruit_tn1/loop_body/Tensordot/Reshape:output:00fruit_tn1/loop_body/Tensordot/Reshape_1:output:0*
T0*
_output_shapes
:	А|
#fruit_tn1/loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            Є
fruit_tn1/loop_body/TensordotReshape.fruit_tn1/loop_body/Tensordot/MatMul:product:0,fruit_tn1/loop_body/Tensordot/shape:output:0*
T0*'
_output_shapes
:АГ
.fruit_tn1/loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"           
)fruit_tn1/loop_body/Tensordot_1/transpose	Transpose,fruit_tn1/loop_body/ReadVariableOp_2:value:07fruit_tn1/loop_body/Tensordot_1/transpose/perm:output:0*
T0*"
_output_shapes
:~
-fruit_tn1/loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ¬
'fruit_tn1/loop_body/Tensordot_1/ReshapeReshape-fruit_tn1/loop_body/Tensordot_1/transpose:y:06fruit_tn1/loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:А
/fruit_tn1/loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ј
)fruit_tn1/loop_body/Tensordot_1/Reshape_1Reshape&fruit_tn1/loop_body/Tensordot:output:08fruit_tn1/loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А(ј
&fruit_tn1/loop_body/Tensordot_1/MatMulMatMul0fruit_tn1/loop_body/Tensordot_1/Reshape:output:02fruit_tn1/loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	А(В
%fruit_tn1/loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"               ¬
fruit_tn1/loop_body/Tensordot_1Reshape0fruit_tn1/loop_body/Tensordot_1/MatMul:product:0.fruit_tn1/loop_body/Tensordot_1/shape:output:0*
T0*+
_output_shapes
:А~
-fruit_tn1/loop_body/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ¬
'fruit_tn1/loop_body/Tensordot_2/ReshapeReshape,fruit_tn1/loop_body/ReadVariableOp_1:value:06fruit_tn1/loop_body/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes
:	А2Л
.fruit_tn1/loop_body/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ѕ
)fruit_tn1/loop_body/Tensordot_2/transpose	Transpose(fruit_tn1/loop_body/Tensordot_1:output:07fruit_tn1/loop_body/Tensordot_2/transpose/perm:output:0*
T0*+
_output_shapes
:АА
/fruit_tn1/loop_body/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      «
)fruit_tn1/loop_body/Tensordot_2/Reshape_1Reshape-fruit_tn1/loop_body/Tensordot_2/transpose:y:08fruit_tn1/loop_body/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А2њ
&fruit_tn1/loop_body/Tensordot_2/MatMulMatMul0fruit_tn1/loop_body/Tensordot_2/Reshape:output:02fruit_tn1/loop_body/Tensordot_2/Reshape_1:output:0*
T0*
_output_shapes

:z
%fruit_tn1/loop_body/Tensordot_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         є
fruit_tn1/loop_body/Tensordot_2Reshape0fruit_tn1/loop_body/Tensordot_2/MatMul:product:0.fruit_tn1/loop_body/Tensordot_2/shape:output:0*
T0*"
_output_shapes
:w
"fruit_tn1/loop_body/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ѓ
fruit_tn1/loop_body/transpose	Transpose(fruit_tn1/loop_body/Tensordot_2:output:0+fruit_tn1/loop_body/transpose/perm:output:0*
T0*"
_output_shapes
:Ъ
&fruit_tn1/loop_body/add/ReadVariableOpReadVariableOp/fruit_tn1_loop_body_add_readvariableop_resource*"
_output_shapes
:*
dtype0†
fruit_tn1/loop_body/addAddV2!fruit_tn1/loop_body/transpose:y:0.fruit_tn1/loop_body/add/ReadVariableOp:value:0*
T0*"
_output_shapes
:f
fruit_tn1/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Е
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
value	B :§
fruit_tn1/pfor/rangeRange#fruit_tn1/pfor/range/start:output:0fruit_tn1/Max:output:0#fruit_tn1/pfor/range/delta:output:0*#
_output_shapes
:€€€€€€€€€h
&fruit_tn1/loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : i
'fruit_tn1/loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :≤
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
value	B :Є
'fruit_tn1/loop_body/SelectV2/pfor/add_1AddV21fruit_tn1/loop_body/SelectV2/pfor/Rank_2:output:02fruit_tn1/loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: ≥
)fruit_tn1/loop_body/SelectV2/pfor/MaximumMaximum1fruit_tn1/loop_body/SelectV2/pfor/Rank_1:output:0)fruit_tn1/loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: ≥
+fruit_tn1/loop_body/SelectV2/pfor/Maximum_1Maximum+fruit_tn1/loop_body/SelectV2/pfor/add_1:z:0-fruit_tn1/loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: t
'fruit_tn1/loop_body/SelectV2/pfor/ShapeShapefruit_tn1/pfor/range:output:0*
T0*
_output_shapes
:±
%fruit_tn1/loop_body/SelectV2/pfor/subSub/fruit_tn1/loop_body/SelectV2/pfor/Maximum_1:z:01fruit_tn1/loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: y
/fruit_tn1/loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Њ
)fruit_tn1/loop_body/SelectV2/pfor/ReshapeReshape)fruit_tn1/loop_body/SelectV2/pfor/sub:z:08fruit_tn1/loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:v
,fruit_tn1/loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Љ
&fruit_tn1/loop_body/SelectV2/pfor/TileTile5fruit_tn1/loop_body/SelectV2/pfor/Tile/input:output:02fruit_tn1/loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: 
5fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
/fruit_tn1/loop_body/SelectV2/pfor/strided_sliceStridedSlice0fruit_tn1/loop_body/SelectV2/pfor/Shape:output:0>fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack:output:0@fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_1:output:0@fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskБ
7fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Г
9fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:э
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
value	B : Ѕ
(fruit_tn1/loop_body/SelectV2/pfor/concatConcatV28fruit_tn1/loop_body/SelectV2/pfor/strided_slice:output:0/fruit_tn1/loop_body/SelectV2/pfor/Tile:output:0:fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1:output:06fruit_tn1/loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ґ
+fruit_tn1/loop_body/SelectV2/pfor/Reshape_1Reshapefruit_tn1/pfor/range:output:01fruit_tn1/loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:€€€€€€€€€д
*fruit_tn1/loop_body/SelectV2/pfor/SelectV2SelectV2fruit_tn1/loop_body/Greater:z:04fruit_tn1/loop_body/SelectV2/pfor/Reshape_1:output:0'fruit_tn1/loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:€€€€€€€€€q
/fruit_tn1/loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : •
*fruit_tn1/loop_body/GatherV2/pfor/GatherV2GatherV2 max_pooling2d_1/MaxPool:output:03fruit_tn1/loop_body/SelectV2/pfor/SelectV2:output:08fruit_tn1/loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*0
_output_shapes
:€€€€€€€€€Аt
2fruit_tn1/loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :“
0fruit_tn1/loop_body/Tensordot/transpose/pfor/addAddV25fruit_tn1/loop_body/Tensordot/transpose/perm:output:0;fruit_tn1/loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:Ж
<fruit_tn1/loop_body/Tensordot/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: z
8fruit_tn1/loop_body/Tensordot/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ≠
3fruit_tn1/loop_body/Tensordot/transpose/pfor/concatConcatV2Efruit_tn1/loop_body/Tensordot/transpose/pfor/concat/values_0:output:04fruit_tn1/loop_body/Tensordot/transpose/pfor/add:z:0Afruit_tn1/loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:с
6fruit_tn1/loop_body/Tensordot/transpose/pfor/Transpose	Transpose3fruit_tn1/loop_body/GatherV2/pfor/GatherV2:output:0<fruit_tn1/loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аx
6fruit_tn1/loop_body/Tensordot/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Г
1fruit_tn1/loop_body/Tensordot/Reshape/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:04fruit_tn1/loop_body/Tensordot/Reshape/shape:output:0?fruit_tn1/loop_body/Tensordot/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:м
2fruit_tn1/loop_body/Tensordot/Reshape/pfor/ReshapeReshape:fruit_tn1/loop_body/Tensordot/transpose/pfor/Transpose:y:0:fruit_tn1/loop_body/Tensordot/Reshape/pfor/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€АЪ
/fruit_tn1/loop_body/Tensordot/MatMul/pfor/ShapeShape;fruit_tn1/loop_body/Tensordot/Reshape/pfor/Reshape:output:0*
T0*
_output_shapes
:{
9fruit_tn1/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ш
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
valueB џ
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
valueB я
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
valueB я
3fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2Reshape8fruit_tn1/loop_body/Tensordot/MatMul/pfor/split:output:2Dfruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ѕ
-fruit_tn1/loop_body/Tensordot/MatMul/pfor/mulMul:fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape:output:0<fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: а
9fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack1fruit_tn1/loop_body/Tensordot/MatMul/pfor/mul:z:0<fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:ъ
3fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape;fruit_tn1/loop_body/Tensordot/Reshape/pfor/Reshape:output:0Bfruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€№
0fruit_tn1/loop_body/Tensordot/MatMul/pfor/MatMulMatMul<fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3:output:00fruit_tn1/loop_body/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
;fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ѓ
9fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePack:fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape:output:0<fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0Dfruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:х
3fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4Reshape:fruit_tn1/loop_body/Tensordot/MatMul/pfor/MatMul:product:0Bfruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€Аp
.fruit_tn1/loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : л
)fruit_tn1/loop_body/Tensordot/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:0,fruit_tn1/loop_body/Tensordot/shape:output:07fruit_tn1/loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ж
*fruit_tn1/loop_body/Tensordot/pfor/ReshapeReshape<fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4:output:02fruit_tn1/loop_body/Tensordot/pfor/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€А|
:fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : П
5fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:08fruit_tn1/loop_body/Tensordot_1/Reshape_1/shape:output:0Cfruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:н
6fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape3fruit_tn1/loop_body/Tensordot/pfor/Reshape:output:0>fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А(†
1fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/ShapeShape?fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:Й
?fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:≠
9fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSlice:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Hfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSlice:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskй
3fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/MinimumMinimumBfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0Dfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: u
3fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :“
1fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/EqualEqual7fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Minimum:z:0<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: Й
4fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          Й
4fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
2fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/SelectSelect5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Equal:z:0=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/t:output:0=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:Л
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSlice:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: Н
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSlice:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSlice:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskє
1fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/stackPackDfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0Dfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0Dfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:И
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose	Transpose?fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€м
3fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/ReshapeReshape9fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose:y:0:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:€€€€€€€€€А(Я
3fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape_1Shape<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:}
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : А
1fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/splitSplitDfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split/split_dim:output:0<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split~
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1Reshape:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split:output:0Ffruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2Reshape:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split:output:1Ffruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3Reshape:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split:output:2Ffruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: „
/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/mulMul>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ж
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePack>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:03fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:€
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4Reshape<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0Dfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€а
2fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul0fruit_tn1/loop_body/Tensordot_1/Reshape:output:0>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€И
=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€є
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePackFfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:0>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:М
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5Reshape<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0Dfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€С
<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          В
7fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1	Transpose>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0Efruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€А(r
0fruit_tn1/loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : с
+fruit_tn1/loop_body/Tensordot_1/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:0.fruit_tn1/loop_body/Tensordot_1/shape:output:09fruit_tn1/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:н
,fruit_tn1/loop_body/Tensordot_1/pfor/ReshapeReshape;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:04fruit_tn1/loop_body/Tensordot_1/pfor/concat:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€Аv
4fruit_tn1/loop_body/Tensordot_2/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ў
2fruit_tn1/loop_body/Tensordot_2/transpose/pfor/addAddV27fruit_tn1/loop_body/Tensordot_2/transpose/perm:output:0=fruit_tn1/loop_body/Tensordot_2/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:И
>fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: |
:fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : µ
5fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concatConcatV2Gfruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/values_0:output:06fruit_tn1/loop_body/Tensordot_2/transpose/pfor/add:z:0Cfruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:€
8fruit_tn1/loop_body/Tensordot_2/transpose/pfor/Transpose	Transpose5fruit_tn1/loop_body/Tensordot_1/pfor/Reshape:output:0>fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€А|
:fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : П
5fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:08fruit_tn1/loop_body/Tensordot_2/Reshape_1/shape:output:0Cfruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ц
6fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/ReshapeReshape<fruit_tn1/loop_body/Tensordot_2/transpose/pfor/Transpose:y:0>fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2†
1fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/ShapeShape?fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:Й
?fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:≠
9fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_sliceStridedSlice:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Hfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1StridedSlice:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskй
3fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/MinimumMinimumBfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice:output:0Dfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: u
3fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :“
1fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/EqualEqual7fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Minimum:z:0<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: Й
4fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          Й
4fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
2fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/SelectSelect5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Equal:z:0=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/t:output:0=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:Л
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2StridedSlice:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: Н
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3StridedSlice:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4StridedSlice:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskє
1fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/stackPackDfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2:output:0Dfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3:output:0Dfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:И
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose	Transpose?fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€м
3fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/ReshapeReshape9fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose:y:0:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:А2€€€€€€€€€Я
3fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape_1Shape<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:}
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : А
1fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/splitSplitDfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split/split_dim:output:0<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split~
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1Reshape:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split:output:0Ffruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2Reshape:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split:output:1Ffruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3Reshape:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split:output:2Ffruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: „
/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/mulMul>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ж
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shapePack>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1:output:03fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:€
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4Reshape<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0Dfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€а
2fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/MatMulMatMul0fruit_tn1/loop_body/Tensordot_2/Reshape:output:0>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€И
=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€є
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shapePackFfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0:output:0>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:М
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5Reshape<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/MatMul:product:0Dfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€С
<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Б
7fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1	Transpose>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5:output:0Efruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€r
0fruit_tn1/loop_body/Tensordot_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : с
+fruit_tn1/loop_body/Tensordot_2/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:0.fruit_tn1/loop_body/Tensordot_2/shape:output:09fruit_tn1/loop_body/Tensordot_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:д
,fruit_tn1/loop_body/Tensordot_2/pfor/ReshapeReshape;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1:y:04fruit_tn1/loop_body/Tensordot_2/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€j
(fruit_tn1/loop_body/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :і
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
value	B : Е
)fruit_tn1/loop_body/transpose/pfor/concatConcatV2;fruit_tn1/loop_body/transpose/pfor/concat/values_0:output:0*fruit_tn1/loop_body/transpose/pfor/add:z:07fruit_tn1/loop_body/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ё
,fruit_tn1/loop_body/transpose/pfor/Transpose	Transpose5fruit_tn1/loop_body/Tensordot_2/pfor/Reshape:output:02fruit_tn1/loop_body/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€c
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
value	B :•
 fruit_tn1/loop_body/add/pfor/addAddV2,fruit_tn1/loop_body/add/pfor/Rank_1:output:0+fruit_tn1/loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: Ґ
$fruit_tn1/loop_body/add/pfor/MaximumMaximum$fruit_tn1/loop_body/add/pfor/add:z:0*fruit_tn1/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: В
"fruit_tn1/loop_body/add/pfor/ShapeShape0fruit_tn1/loop_body/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:Ю
 fruit_tn1/loop_body/add/pfor/subSub(fruit_tn1/loop_body/add/pfor/Maximum:z:0*fruit_tn1/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: t
*fruit_tn1/loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:ѓ
$fruit_tn1/loop_body/add/pfor/ReshapeReshape$fruit_tn1/loop_body/add/pfor/sub:z:03fruit_tn1/loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:q
'fruit_tn1/loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:≠
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
valueB:а
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
valueB:ж
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
value	B : ®
#fruit_tn1/loop_body/add/pfor/concatConcatV23fruit_tn1/loop_body/add/pfor/strided_slice:output:0*fruit_tn1/loop_body/add/pfor/Tile:output:05fruit_tn1/loop_body/add/pfor/strided_slice_1:output:01fruit_tn1/loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ћ
&fruit_tn1/loop_body/add/pfor/Reshape_1Reshape0fruit_tn1/loop_body/transpose/pfor/Transpose:y:0,fruit_tn1/loop_body/add/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€∆
"fruit_tn1/loop_body/add/pfor/AddV2AddV2/fruit_tn1/loop_body/add/pfor/Reshape_1:output:0.fruit_tn1/loop_body/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€h
fruit_tn1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   Ш
fruit_tn1/ReshapeReshape&fruit_tn1/loop_body/add/pfor/AddV2:z:0 fruit_tn1/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@d
fruit_tn1/ReluRelufruit_tn1/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@~
$fruit_tn1/ActivityRegularizer/SquareSquarefruit_tn1/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@t
#fruit_tn1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       °
!fruit_tn1/ActivityRegularizer/SumSum(fruit_tn1/ActivityRegularizer/Square:y:0,fruit_tn1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: h
#fruit_tn1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ђ≈'7£
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
valueB:з
+fruit_tn1/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn1/ActivityRegularizer/Shape:output:0:fruit_tn1/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
"fruit_tn1/ActivityRegularizer/CastCast4fruit_tn1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: †
%fruit_tn1/ActivityRegularizer/truedivRealDiv%fruit_tn1/ActivityRegularizer/mul:z:0&fruit_tn1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: [
dropout1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @М
dropout1/dropout/MulMulfruit_tn1/Relu:activations:0dropout1/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@b
dropout1/dropout/ShapeShapefruit_tn1/Relu:activations:0*
T0*
_output_shapes
:Ю
-dropout1/dropout/random_uniform/RandomUniformRandomUniformdropout1/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0d
dropout1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ѕ
dropout1/dropout/GreaterEqualGreaterEqual6dropout1/dropout/random_uniform/RandomUniform:output:0(dropout1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Б
dropout1/dropout/CastCast!dropout1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Д
dropout1/dropout/Mul_1Muldropout1/dropout/Mul:z:0dropout1/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@Y
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
valueB:Г
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
value	B :Н
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
value	B : ±
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
valueB:µ
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
value	B :Ы
fruit_tn2/loop_body/GreaterGreater*fruit_tn2/loop_body/strided_slice:output:0&fruit_tn2/loop_body/Greater/y:output:0*
T0*
_output_shapes
: `
fruit_tn2/loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : »
fruit_tn2/loop_body/SelectV2SelectV2fruit_tn2/loop_body/Greater:z:03fruit_tn2/loop_body/PlaceholderWithDefault:output:0'fruit_tn2/loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: c
!fruit_tn2/loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : я
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
valueB"         ¶
fruit_tn2/loop_body/ReshapeReshape%fruit_tn2/loop_body/GatherV2:output:0*fruit_tn2/loop_body/Reshape/shape:output:0*
T0*"
_output_shapes
:О
"fruit_tn2/loop_body/ReadVariableOpReadVariableOp+fruit_tn2_loop_body_readvariableop_resource*
_output_shapes

:*
dtype0Ы
$fruit_tn2/loop_body/ReadVariableOp_1ReadVariableOp-fruit_tn2_loop_body_readvariableop_1_resource*'
_output_shapes
:Г*
dtype0Т
$fruit_tn2/loop_body/ReadVariableOp_2ReadVariableOp-fruit_tn2_loop_body_readvariableop_2_resource*
_output_shapes

:*
dtype0Б
,fruit_tn2/loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Њ
'fruit_tn2/loop_body/Tensordot/transpose	Transpose$fruit_tn2/loop_body/Reshape:output:05fruit_tn2/loop_body/Tensordot/transpose/perm:output:0*
T0*"
_output_shapes
:|
+fruit_tn2/loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Љ
%fruit_tn2/loop_body/Tensordot/ReshapeReshape+fruit_tn2/loop_body/Tensordot/transpose:y:04fruit_tn2/loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:≥
$fruit_tn2/loop_body/Tensordot/MatMulMatMul.fruit_tn2/loop_body/Tensordot/Reshape:output:0*fruit_tn2/loop_body/ReadVariableOp:value:0*
T0*
_output_shapes

:x
#fruit_tn2/loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ≥
fruit_tn2/loop_body/TensordotReshape.fruit_tn2/loop_body/Tensordot/MatMul:product:0,fruit_tn2/loop_body/Tensordot/shape:output:0*
T0*"
_output_shapes
:З
.fruit_tn2/loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ѕ
)fruit_tn2/loop_body/Tensordot_1/transpose	Transpose,fruit_tn2/loop_body/ReadVariableOp_1:value:07fruit_tn2/loop_body/Tensordot_1/transpose/perm:output:0*
T0*'
_output_shapes
:Г~
-fruit_tn2/loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"     √
'fruit_tn2/loop_body/Tensordot_1/ReshapeReshape-fruit_tn2/loop_body/Tensordot_1/transpose:y:06fruit_tn2/loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes
:	МЕ
0fruit_tn2/loop_body/Tensordot_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          »
+fruit_tn2/loop_body/Tensordot_1/transpose_1	Transpose&fruit_tn2/loop_body/Tensordot:output:09fruit_tn2/loop_body/Tensordot_1/transpose_1/perm:output:0*
T0*"
_output_shapes
:А
/fruit_tn2/loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      »
)fruit_tn2/loop_body/Tensordot_1/Reshape_1Reshape/fruit_tn2/loop_body/Tensordot_1/transpose_1:y:08fruit_tn2/loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:ј
&fruit_tn2/loop_body/Tensordot_1/MatMulMatMul0fruit_tn2/loop_body/Tensordot_1/Reshape:output:02fruit_tn2/loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	Мz
%fruit_tn2/loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"Г         Ї
fruit_tn2/loop_body/Tensordot_1Reshape0fruit_tn2/loop_body/Tensordot_1/MatMul:product:0.fruit_tn2/loop_body/Tensordot_1/shape:output:0*
T0*#
_output_shapes
:Г~
-fruit_tn2/loop_body/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ѕ
'fruit_tn2/loop_body/Tensordot_2/ReshapeReshape,fruit_tn2/loop_body/ReadVariableOp_2:value:06fruit_tn2/loop_body/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes

:Г
.fruit_tn2/loop_body/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
)fruit_tn2/loop_body/Tensordot_2/transpose	Transpose(fruit_tn2/loop_body/Tensordot_1:output:07fruit_tn2/loop_body/Tensordot_2/transpose/perm:output:0*
T0*#
_output_shapes
:ГА
/fruit_tn2/loop_body/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   Г   «
)fruit_tn2/loop_body/Tensordot_2/Reshape_1Reshape-fruit_tn2/loop_body/Tensordot_2/transpose:y:08fruit_tn2/loop_body/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	Гј
&fruit_tn2/loop_body/Tensordot_2/MatMulMatMul0fruit_tn2/loop_body/Tensordot_2/Reshape:output:02fruit_tn2/loop_body/Tensordot_2/Reshape_1:output:0*
T0*
_output_shapes
:	Гp
%fruit_tn2/loop_body/Tensordot_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:Г≤
fruit_tn2/loop_body/Tensordot_2Reshape0fruit_tn2/loop_body/Tensordot_2/MatMul:product:0.fruit_tn2/loop_body/Tensordot_2/shape:output:0*
T0*
_output_shapes	
:ГУ
&fruit_tn2/loop_body/add/ReadVariableOpReadVariableOp/fruit_tn2_loop_body_add_readvariableop_resource*
_output_shapes	
:Г*
dtype0†
fruit_tn2/loop_body/addAddV2(fruit_tn2/loop_body/Tensordot_2:output:0.fruit_tn2/loop_body/add/ReadVariableOp:value:0*
T0*
_output_shapes	
:Гf
fruit_tn2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Е
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
value	B :§
fruit_tn2/pfor/rangeRange#fruit_tn2/pfor/range/start:output:0fruit_tn2/Max:output:0#fruit_tn2/pfor/range/delta:output:0*#
_output_shapes
:€€€€€€€€€h
&fruit_tn2/loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : i
'fruit_tn2/loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :≤
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
value	B :Є
'fruit_tn2/loop_body/SelectV2/pfor/add_1AddV21fruit_tn2/loop_body/SelectV2/pfor/Rank_2:output:02fruit_tn2/loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: ≥
)fruit_tn2/loop_body/SelectV2/pfor/MaximumMaximum1fruit_tn2/loop_body/SelectV2/pfor/Rank_1:output:0)fruit_tn2/loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: ≥
+fruit_tn2/loop_body/SelectV2/pfor/Maximum_1Maximum+fruit_tn2/loop_body/SelectV2/pfor/add_1:z:0-fruit_tn2/loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: t
'fruit_tn2/loop_body/SelectV2/pfor/ShapeShapefruit_tn2/pfor/range:output:0*
T0*
_output_shapes
:±
%fruit_tn2/loop_body/SelectV2/pfor/subSub/fruit_tn2/loop_body/SelectV2/pfor/Maximum_1:z:01fruit_tn2/loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: y
/fruit_tn2/loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Њ
)fruit_tn2/loop_body/SelectV2/pfor/ReshapeReshape)fruit_tn2/loop_body/SelectV2/pfor/sub:z:08fruit_tn2/loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:v
,fruit_tn2/loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Љ
&fruit_tn2/loop_body/SelectV2/pfor/TileTile5fruit_tn2/loop_body/SelectV2/pfor/Tile/input:output:02fruit_tn2/loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: 
5fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
/fruit_tn2/loop_body/SelectV2/pfor/strided_sliceStridedSlice0fruit_tn2/loop_body/SelectV2/pfor/Shape:output:0>fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack:output:0@fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_1:output:0@fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskБ
7fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Г
9fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:э
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
value	B : Ѕ
(fruit_tn2/loop_body/SelectV2/pfor/concatConcatV28fruit_tn2/loop_body/SelectV2/pfor/strided_slice:output:0/fruit_tn2/loop_body/SelectV2/pfor/Tile:output:0:fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1:output:06fruit_tn2/loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ґ
+fruit_tn2/loop_body/SelectV2/pfor/Reshape_1Reshapefruit_tn2/pfor/range:output:01fruit_tn2/loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:€€€€€€€€€д
*fruit_tn2/loop_body/SelectV2/pfor/SelectV2SelectV2fruit_tn2/loop_body/Greater:z:04fruit_tn2/loop_body/SelectV2/pfor/Reshape_1:output:0'fruit_tn2/loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:€€€€€€€€€q
/fruit_tn2/loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ц
*fruit_tn2/loop_body/GatherV2/pfor/GatherV2GatherV2dropout1/dropout/Mul_1:z:03fruit_tn2/loop_body/SelectV2/pfor/SelectV2:output:08fruit_tn2/loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:€€€€€€€€€@n
,fruit_tn2/loop_body/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : е
'fruit_tn2/loop_body/Reshape/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:0*fruit_tn2/loop_body/Reshape/shape:output:05fruit_tn2/loop_body/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:‘
(fruit_tn2/loop_body/Reshape/pfor/ReshapeReshape3fruit_tn2/loop_body/GatherV2/pfor/GatherV2:output:00fruit_tn2/loop_body/Reshape/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€t
2fruit_tn2/loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :“
0fruit_tn2/loop_body/Tensordot/transpose/pfor/addAddV25fruit_tn2/loop_body/Tensordot/transpose/perm:output:0;fruit_tn2/loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:Ж
<fruit_tn2/loop_body/Tensordot/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: z
8fruit_tn2/loop_body/Tensordot/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ≠
3fruit_tn2/loop_body/Tensordot/transpose/pfor/concatConcatV2Efruit_tn2/loop_body/Tensordot/transpose/pfor/concat/values_0:output:04fruit_tn2/loop_body/Tensordot/transpose/pfor/add:z:0Afruit_tn2/loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:о
6fruit_tn2/loop_body/Tensordot/transpose/pfor/Transpose	Transpose1fruit_tn2/loop_body/Reshape/pfor/Reshape:output:0<fruit_tn2/loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€x
6fruit_tn2/loop_body/Tensordot/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Г
1fruit_tn2/loop_body/Tensordot/Reshape/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:04fruit_tn2/loop_body/Tensordot/Reshape/shape:output:0?fruit_tn2/loop_body/Tensordot/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:л
2fruit_tn2/loop_body/Tensordot/Reshape/pfor/ReshapeReshape:fruit_tn2/loop_body/Tensordot/transpose/pfor/Transpose:y:0:fruit_tn2/loop_body/Tensordot/Reshape/pfor/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€Ъ
/fruit_tn2/loop_body/Tensordot/MatMul/pfor/ShapeShape;fruit_tn2/loop_body/Tensordot/Reshape/pfor/Reshape:output:0*
T0*
_output_shapes
:{
9fruit_tn2/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ш
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
valueB џ
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
valueB я
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
valueB я
3fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2Reshape8fruit_tn2/loop_body/Tensordot/MatMul/pfor/split:output:2Dfruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ѕ
-fruit_tn2/loop_body/Tensordot/MatMul/pfor/mulMul:fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape:output:0<fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: а
9fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack1fruit_tn2/loop_body/Tensordot/MatMul/pfor/mul:z:0<fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:ъ
3fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape;fruit_tn2/loop_body/Tensordot/Reshape/pfor/Reshape:output:0Bfruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€÷
0fruit_tn2/loop_body/Tensordot/MatMul/pfor/MatMulMatMul<fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0*fruit_tn2/loop_body/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
;fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ѓ
9fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePack:fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape:output:0<fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0Dfruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:ф
3fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4Reshape:fruit_tn2/loop_body/Tensordot/MatMul/pfor/MatMul:product:0Bfruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€p
.fruit_tn2/loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : л
)fruit_tn2/loop_body/Tensordot/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:0,fruit_tn2/loop_body/Tensordot/shape:output:07fruit_tn2/loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:б
*fruit_tn2/loop_body/Tensordot/pfor/ReshapeReshape<fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4:output:02fruit_tn2/loop_body/Tensordot/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€x
6fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :ё
4fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/addAddV29fruit_tn2/loop_body/Tensordot_1/transpose_1/perm:output:0?fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/add/y:output:0*
T0*
_output_shapes
:К
@fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: ~
<fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : љ
7fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concatConcatV2Ifruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/values_0:output:08fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/add:z:0Efruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ш
:fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/Transpose	Transpose3fruit_tn2/loop_body/Tensordot/pfor/Reshape:output:0@fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€|
:fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : П
5fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:08fruit_tn2/loop_body/Tensordot_1/Reshape_1/shape:output:0Cfruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ч
6fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape>fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/Transpose:y:0>fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€†
1fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/ShapeShape?fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:Й
?fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:≠
9fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSlice:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Hfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSlice:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskй
3fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/MinimumMinimumBfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0Dfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: u
3fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :“
1fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/EqualEqual7fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Minimum:z:0<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: Й
4fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          Й
4fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
2fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/SelectSelect5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Equal:z:0=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/t:output:0=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:Л
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSlice:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: Н
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSlice:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSlice:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskє
1fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/stackPackDfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0Dfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0Dfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:И
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose	Transpose?fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€л
3fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/ReshapeReshape9fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose:y:0:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*+
_output_shapes
:€€€€€€€€€Я
3fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape_1Shape<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:}
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : А
1fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/splitSplitDfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split/split_dim:output:0<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split~
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1Reshape:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split:output:0Ffruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2Reshape:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split:output:1Ffruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3Reshape:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split:output:2Ffruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: „
/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/mulMul>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ж
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePack>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:03fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:€
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4Reshape<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0Dfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€б
2fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul0fruit_tn2/loop_body/Tensordot_1/Reshape:output:0>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*(
_output_shapes
:М€€€€€€€€€И
=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€є
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePackFfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:0>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:М
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5Reshape<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0Dfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€С
<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          В
7fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1	Transpose>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0Efruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Мr
0fruit_tn2/loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : с
+fruit_tn2/loop_body/Tensordot_1/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:0.fruit_tn2/loop_body/Tensordot_1/shape:output:09fruit_tn2/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:е
,fruit_tn2/loop_body/Tensordot_1/pfor/ReshapeReshape;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:04fruit_tn2/loop_body/Tensordot_1/pfor/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€Гv
4fruit_tn2/loop_body/Tensordot_2/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ў
2fruit_tn2/loop_body/Tensordot_2/transpose/pfor/addAddV27fruit_tn2/loop_body/Tensordot_2/transpose/perm:output:0=fruit_tn2/loop_body/Tensordot_2/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:И
>fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: |
:fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : µ
5fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concatConcatV2Gfruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/values_0:output:06fruit_tn2/loop_body/Tensordot_2/transpose/pfor/add:z:0Cfruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ч
8fruit_tn2/loop_body/Tensordot_2/transpose/pfor/Transpose	Transpose5fruit_tn2/loop_body/Tensordot_1/pfor/Reshape:output:0>fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€Г|
:fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : П
5fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:08fruit_tn2/loop_body/Tensordot_2/Reshape_1/shape:output:0Cfruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ц
6fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/ReshapeReshape<fruit_tn2/loop_body/Tensordot_2/transpose/pfor/Transpose:y:0>fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Г†
1fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/ShapeShape?fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:Й
?fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:≠
9fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_sliceStridedSlice:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Hfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1StridedSlice:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskй
3fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/MinimumMinimumBfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice:output:0Dfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: u
3fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :“
1fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/EqualEqual7fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Minimum:z:0<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: Й
4fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          Й
4fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
2fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/SelectSelect5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Equal:z:0=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/t:output:0=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:Л
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2StridedSlice:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: Н
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3StridedSlice:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4StridedSlice:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskє
1fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/stackPackDfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2:output:0Dfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3:output:0Dfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:И
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose	Transpose?fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€м
3fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/ReshapeReshape9fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose:y:0:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:€€€€€€€€€ГЯ
3fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape_1Shape<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:}
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : А
1fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/splitSplitDfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split/split_dim:output:0<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split~
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1Reshape:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split:output:0Ffruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2Reshape:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split:output:1Ffruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3Reshape:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split:output:2Ffruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: „
/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/mulMul>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ж
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shapePack>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1:output:03fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:€
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4Reshape<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0Dfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€а
2fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/MatMulMatMul0fruit_tn2/loop_body/Tensordot_2/Reshape:output:0>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€И
=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€є
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shapePackFfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0:output:0>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:М
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5Reshape<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/MatMul:product:0Dfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€С
<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          В
7fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1	Transpose>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5:output:0Efruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Гr
0fruit_tn2/loop_body/Tensordot_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : с
+fruit_tn2/loop_body/Tensordot_2/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:0.fruit_tn2/loop_body/Tensordot_2/shape:output:09fruit_tn2/loop_body/Tensordot_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ё
,fruit_tn2/loop_body/Tensordot_2/pfor/ReshapeReshape;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1:y:04fruit_tn2/loop_body/Tensordot_2/pfor/concat:output:0*
T0*(
_output_shapes
:€€€€€€€€€Гc
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
value	B :•
 fruit_tn2/loop_body/add/pfor/addAddV2,fruit_tn2/loop_body/add/pfor/Rank_1:output:0+fruit_tn2/loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: Ґ
$fruit_tn2/loop_body/add/pfor/MaximumMaximum$fruit_tn2/loop_body/add/pfor/add:z:0*fruit_tn2/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: З
"fruit_tn2/loop_body/add/pfor/ShapeShape5fruit_tn2/loop_body/Tensordot_2/pfor/Reshape:output:0*
T0*
_output_shapes
:Ю
 fruit_tn2/loop_body/add/pfor/subSub(fruit_tn2/loop_body/add/pfor/Maximum:z:0*fruit_tn2/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: t
*fruit_tn2/loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:ѓ
$fruit_tn2/loop_body/add/pfor/ReshapeReshape$fruit_tn2/loop_body/add/pfor/sub:z:03fruit_tn2/loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:q
'fruit_tn2/loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:≠
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
valueB:а
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
valueB:ж
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
value	B : ®
#fruit_tn2/loop_body/add/pfor/concatConcatV23fruit_tn2/loop_body/add/pfor/strided_slice:output:0*fruit_tn2/loop_body/add/pfor/Tile:output:05fruit_tn2/loop_body/add/pfor/strided_slice_1:output:01fruit_tn2/loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:…
&fruit_tn2/loop_body/add/pfor/Reshape_1Reshape5fruit_tn2/loop_body/Tensordot_2/pfor/Reshape:output:0,fruit_tn2/loop_body/add/pfor/concat:output:0*
T0*(
_output_shapes
:€€€€€€€€€Гњ
"fruit_tn2/loop_body/add/pfor/AddV2AddV2/fruit_tn2/loop_body/add/pfor/Reshape_1:output:0.fruit_tn2/loop_body/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Гh
fruit_tn2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Г   Щ
fruit_tn2/ReshapeReshape&fruit_tn2/loop_body/add/pfor/AddV2:z:0 fruit_tn2/Reshape/shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€Гk
fruit_tn2/SoftmaxSoftmaxfruit_tn2/Reshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€Г~
$fruit_tn2/ActivityRegularizer/SquareSquarefruit_tn2/Softmax:softmax:0*
T0*(
_output_shapes
:€€€€€€€€€Гt
#fruit_tn2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       °
!fruit_tn2/ActivityRegularizer/SumSum(fruit_tn2/ActivityRegularizer/Square:y:0,fruit_tn2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: h
#fruit_tn2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ђ≈'7£
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
valueB:з
+fruit_tn2/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn2/ActivityRegularizer/Shape:output:0:fruit_tn2/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
"fruit_tn2/ActivityRegularizer/CastCast4fruit_tn2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: †
%fruit_tn2/ActivityRegularizer/truedivRealDiv%fruit_tn2/ActivityRegularizer/mul:z:0&fruit_tn2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: k
IdentityIdentityfruit_tn2/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Гi

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
: ї
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
?:€€€€€€€€€22: : : : : : : : : : : : : : : : : : 2>
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
:€€€€€€€€€22
 
_user_specified_nameinputs
¬Ъ
≠
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
-conv2dmpo_1_reshape_2_readvariableop_resource:	АA
+fruit_tn1_loop_body_readvariableop_resource:H
-fruit_tn1_loop_body_readvariableop_1_resource:АC
-fruit_tn1_loop_body_readvariableop_2_resource:E
/fruit_tn1_loop_body_add_readvariableop_resource:=
+fruit_tn2_loop_body_readvariableop_resource:H
-fruit_tn2_loop_body_readvariableop_1_resource:Г?
-fruit_tn2_loop_body_readvariableop_2_resource:>
/fruit_tn2_loop_body_add_readvariableop_resource:	Г
identity

identity_1

identity_2

identity_3

identity_4ИҐconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOpҐconv2dmpo/ReadVariableOpҐconv2dmpo/ReadVariableOp_1Ґ"conv2dmpo/Reshape_2/ReadVariableOpҐconv2dmpo_1/ReadVariableOpҐconv2dmpo_1/ReadVariableOp_1Ґconv2dmpo_1/ReadVariableOp_2Ґconv2dmpo_1/ReadVariableOp_3Ґ$conv2dmpo_1/Reshape_2/ReadVariableOpҐ"fruit_tn1/loop_body/ReadVariableOpҐ$fruit_tn1/loop_body/ReadVariableOp_1Ґ$fruit_tn1/loop_body/ReadVariableOp_2Ґ&fruit_tn1/loop_body/add/ReadVariableOpҐ"fruit_tn2/loop_body/ReadVariableOpҐ$fruit_tn2/loop_body/ReadVariableOp_1Ґ$fruit_tn2/loop_body/ReadVariableOp_2Ґ&fruit_tn2/loop_body/add/ReadVariableOpК
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0®
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€//*
paddingVALID*
strides
А
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Т
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€//В
conv2dmpo/ReadVariableOpReadVariableOp!conv2dmpo_readvariableop_resource*&
_output_shapes
:*
dtype0Ж
conv2dmpo/ReadVariableOp_1ReadVariableOp#conv2dmpo_readvariableop_1_resource*&
_output_shapes
:*
dtype0{
"conv2dmpo/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ™
conv2dmpo/Tensordot/transpose	Transpose conv2dmpo/ReadVariableOp:value:0+conv2dmpo/Tensordot/transpose/perm:output:0*
T0*&
_output_shapes
:r
!conv2dmpo/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ю
conv2dmpo/Tensordot/ReshapeReshape!conv2dmpo/Tensordot/transpose:y:0*conv2dmpo/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:}
$conv2dmpo/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ∞
conv2dmpo/Tensordot/transpose_1	Transpose"conv2dmpo/ReadVariableOp_1:value:0-conv2dmpo/Tensordot/transpose_1/perm:output:0*
T0*&
_output_shapes
:t
#conv2dmpo/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      §
conv2dmpo/Tensordot/Reshape_1Reshape#conv2dmpo/Tensordot/transpose_1:y:0,conv2dmpo/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:Ы
conv2dmpo/Tensordot/MatMulMatMul$conv2dmpo/Tensordot/Reshape:output:0&conv2dmpo/Tensordot/Reshape_1:output:0*
T0*
_output_shapes

:z
conv2dmpo/Tensordot/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  °
conv2dmpo/TensordotReshape$conv2dmpo/Tensordot/MatMul:product:0"conv2dmpo/Tensordot/shape:output:0*
T0*.
_output_shapes
:y
conv2dmpo/transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   Ъ
conv2dmpo/transpose	Transposeconv2dmpo/Tensordot:output:0!conv2dmpo/transpose/perm:output:0*
T0*.
_output_shapes
:{
conv2dmpo/transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   Щ
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
valueB:€
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
valueB:Й
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
€€€€€€€€€≤
conv2dmpo/concatConcatV2"conv2dmpo/strided_slice_1:output:0"conv2dmpo/concat/values_1:output:0conv2dmpo/concat/axis:output:0*
N*
T0*
_output_shapes
:З
conv2dmpo/ReshapeReshapeconv2dmpo/transpose_1:y:0conv2dmpo/concat:output:0*
T0**
_output_shapes
:w
conv2dmpo/transpose_2/permConst*
_output_shapes
:*
dtype0*)
value B"                Ш
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
valueB:Й
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
valueB:Л
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
€€€€€€€€€Є
conv2dmpo/concat_1ConcatV2"conv2dmpo/strided_slice_3:output:0$conv2dmpo/concat_1/values_1:output:0 conv2dmpo/concat_1/axis:output:0*
N*
T0*
_output_shapes
:З
conv2dmpo/Reshape_1Reshapeconv2dmpo/transpose_2:y:0conv2dmpo/concat_1:output:0*
T0*&
_output_shapes
:≥
conv2dmpo/Conv2DConv2Dconv2d/BiasAdd:output:0conv2dmpo/Reshape_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€//*
paddingSAME*
strides
К
"conv2dmpo/Reshape_2/ReadVariableOpReadVariableOp+conv2dmpo_reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0j
conv2dmpo/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ч
conv2dmpo/Reshape_2Reshape*conv2dmpo/Reshape_2/ReadVariableOp:value:0"conv2dmpo/Reshape_2/shape:output:0*
T0*
_output_shapes

:Й
conv2dmpo/addAddV2conv2dmpo/Conv2D:output:0conv2dmpo/Reshape_2:output:0*
T0*/
_output_shapes
:€€€€€€€€€//c
conv2dmpo/ReluReluconv2dmpo/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€//Ж
$conv2dmpo/ActivityRegularizer/SquareSquareconv2dmpo/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€//|
#conv2dmpo/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             °
!conv2dmpo/ActivityRegularizer/SumSum(conv2dmpo/ActivityRegularizer/Square:y:0,conv2dmpo/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: h
#conv2dmpo/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ђ≈'7£
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
valueB:з
+conv2dmpo/ActivityRegularizer/strided_sliceStridedSlice,conv2dmpo/ActivityRegularizer/Shape:output:0:conv2dmpo/ActivityRegularizer/strided_slice/stack:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_1:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
"conv2dmpo/ActivityRegularizer/CastCast4conv2dmpo/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: †
%conv2dmpo/ActivityRegularizer/truedivRealDiv%conv2dmpo/ActivityRegularizer/mul:z:0&conv2dmpo/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Ђ
max_pooling2d/MaxPoolMaxPoolconv2dmpo/Relu:activations:0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
Ж
conv2dmpo_1/ReadVariableOpReadVariableOp#conv2dmpo_1_readvariableop_resource*&
_output_shapes
:*
dtype0К
conv2dmpo_1/ReadVariableOp_1ReadVariableOp%conv2dmpo_1_readvariableop_1_resource*&
_output_shapes
:*
dtype0К
conv2dmpo_1/ReadVariableOp_2ReadVariableOp%conv2dmpo_1_readvariableop_2_resource*&
_output_shapes
:*
dtype0К
conv2dmpo_1/ReadVariableOp_3ReadVariableOp%conv2dmpo_1_readvariableop_3_resource*&
_output_shapes
:*
dtype0}
$conv2dmpo_1/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ≤
conv2dmpo_1/Tensordot/transpose	Transpose$conv2dmpo_1/ReadVariableOp_2:value:0-conv2dmpo_1/Tensordot/transpose/perm:output:0*
T0*&
_output_shapes
:t
#conv2dmpo_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@      §
conv2dmpo_1/Tensordot/ReshapeReshape#conv2dmpo_1/Tensordot/transpose:y:0,conv2dmpo_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@
&conv2dmpo_1/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ґ
!conv2dmpo_1/Tensordot/transpose_1	Transpose$conv2dmpo_1/ReadVariableOp_3:value:0/conv2dmpo_1/Tensordot/transpose_1/perm:output:0*
T0*&
_output_shapes
:v
%conv2dmpo_1/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ™
conv2dmpo_1/Tensordot/Reshape_1Reshape%conv2dmpo_1/Tensordot/transpose_1:y:0.conv2dmpo_1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:°
conv2dmpo_1/Tensordot/MatMulMatMul&conv2dmpo_1/Tensordot/Reshape:output:0(conv2dmpo_1/Tensordot/Reshape_1:output:0*
T0*
_output_shapes

:@|
conv2dmpo_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  І
conv2dmpo_1/TensordotReshape&conv2dmpo_1/Tensordot/MatMul:product:0$conv2dmpo_1/Tensordot/shape:output:0*
T0*.
_output_shapes
:
&conv2dmpo_1/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ґ
!conv2dmpo_1/Tensordot_1/transpose	Transpose$conv2dmpo_1/ReadVariableOp_1:value:0/conv2dmpo_1/Tensordot_1/transpose/perm:output:0*
T0*&
_output_shapes
:v
%conv2dmpo_1/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@      ™
conv2dmpo_1/Tensordot_1/ReshapeReshape%conv2dmpo_1/Tensordot_1/transpose:y:0.conv2dmpo_1/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:@Б
(conv2dmpo_1/Tensordot_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             Є
#conv2dmpo_1/Tensordot_1/transpose_1	Transpose"conv2dmpo_1/ReadVariableOp:value:01conv2dmpo_1/Tensordot_1/transpose_1/perm:output:0*
T0*&
_output_shapes
:x
'conv2dmpo_1/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ∞
!conv2dmpo_1/Tensordot_1/Reshape_1Reshape'conv2dmpo_1/Tensordot_1/transpose_1:y:00conv2dmpo_1/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:І
conv2dmpo_1/Tensordot_1/MatMulMatMul(conv2dmpo_1/Tensordot_1/Reshape:output:0*conv2dmpo_1/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes

:@~
conv2dmpo_1/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  ≠
conv2dmpo_1/Tensordot_1Reshape(conv2dmpo_1/Tensordot_1/MatMul:product:0&conv2dmpo_1/Tensordot_1/shape:output:0*
T0*.
_output_shapes
:З
&conv2dmpo_1/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   Є
!conv2dmpo_1/Tensordot_2/transpose	Transposeconv2dmpo_1/Tensordot:output:0/conv2dmpo_1/Tensordot_2/transpose/perm:output:0*
T0*.
_output_shapes
:v
%conv2dmpo_1/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"А      Ђ
conv2dmpo_1/Tensordot_2/ReshapeReshape%conv2dmpo_1/Tensordot_2/transpose:y:0.conv2dmpo_1/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes
:	АЙ
(conv2dmpo_1/Tensordot_2/transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   Њ
#conv2dmpo_1/Tensordot_2/transpose_1	Transpose conv2dmpo_1/Tensordot_1:output:01conv2dmpo_1/Tensordot_2/transpose_1/perm:output:0*
T0*.
_output_shapes
:x
'conv2dmpo_1/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   А   ±
!conv2dmpo_1/Tensordot_2/Reshape_1Reshape'conv2dmpo_1/Tensordot_2/transpose_1:y:00conv2dmpo_1/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А©
conv2dmpo_1/Tensordot_2/MatMulMatMul(conv2dmpo_1/Tensordot_2/Reshape:output:0*conv2dmpo_1/Tensordot_2/Reshape_1:output:0*
T0* 
_output_shapes
:
ААО
conv2dmpo_1/Tensordot_2/shapeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                              љ
conv2dmpo_1/Tensordot_2Reshape(conv2dmpo_1/Tensordot_2/MatMul:product:0&conv2dmpo_1/Tensordot_2/shape:output:0*
T0*>
_output_shapes,
*:(Л
conv2dmpo_1/transpose/permConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                   	            ≤
conv2dmpo_1/transpose	Transpose conv2dmpo_1/Tensordot_2:output:0#conv2dmpo_1/transpose/perm:output:0*
T0*>
_output_shapes,
*:(Н
conv2dmpo_1/transpose_1/permConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                	               ѓ
conv2dmpo_1/transpose_1	Transposeconv2dmpo_1/transpose:y:0%conv2dmpo_1/transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(В
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
valueB:Й
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
valueB:У
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
€€€€€€€€€Ї
conv2dmpo_1/concatConcatV2$conv2dmpo_1/strided_slice_1:output:0$conv2dmpo_1/concat/values_1:output:0 conv2dmpo_1/concat/axis:output:0*
N*
T0*
_output_shapes
:Х
conv2dmpo_1/ReshapeReshapeconv2dmpo_1/transpose_1:y:0conv2dmpo_1/concat:output:0*
T0*2
_output_shapes 
:Б
conv2dmpo_1/transpose_2/permConst*
_output_shapes
:*
dtype0*1
value(B&"                      ¶
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
valueB:У
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
valueB:Х
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
€€€€€€€€€ј
conv2dmpo_1/concat_1ConcatV2$conv2dmpo_1/strided_slice_3:output:0&conv2dmpo_1/concat_1/values_1:output:0"conv2dmpo_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:О
conv2dmpo_1/Reshape_1Reshapeconv2dmpo_1/transpose_2:y:0conv2dmpo_1/concat_1:output:0*
T0*'
_output_shapes
:Ањ
conv2dmpo_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0conv2dmpo_1/Reshape_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
П
$conv2dmpo_1/Reshape_2/ReadVariableOpReadVariableOp-conv2dmpo_1_reshape_2_readvariableop_resource*
_output_shapes	
:А*
dtype0l
conv2dmpo_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ю
conv2dmpo_1/Reshape_2Reshape,conv2dmpo_1/Reshape_2/ReadVariableOp:value:0$conv2dmpo_1/Reshape_2/shape:output:0*
T0*
_output_shapes
:	АР
conv2dmpo_1/addAddV2conv2dmpo_1/Conv2D:output:0conv2dmpo_1/Reshape_2:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аh
conv2dmpo_1/ReluReluconv2dmpo_1/add:z:0*
T0*0
_output_shapes
:€€€€€€€€€АЛ
&conv2dmpo_1/ActivityRegularizer/SquareSquareconv2dmpo_1/Relu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€А~
%conv2dmpo_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             І
#conv2dmpo_1/ActivityRegularizer/SumSum*conv2dmpo_1/ActivityRegularizer/Square:y:0.conv2dmpo_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: j
%conv2dmpo_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ђ≈'7©
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
valueB:с
-conv2dmpo_1/ActivityRegularizer/strided_sliceStridedSlice.conv2dmpo_1/ActivityRegularizer/Shape:output:0<conv2dmpo_1/ActivityRegularizer/strided_slice/stack:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_1:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskФ
$conv2dmpo_1/ActivityRegularizer/CastCast6conv2dmpo_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ¶
'conv2dmpo_1/ActivityRegularizer/truedivRealDiv'conv2dmpo_1/ActivityRegularizer/mul:z:0(conv2dmpo_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ∞
max_pooling2d_1/MaxPoolMaxPoolconv2dmpo_1/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
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
valueB:Г
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
value	B :Н
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
value	B : ±
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
valueB:µ
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
value	B :Ы
fruit_tn1/loop_body/GreaterGreater*fruit_tn1/loop_body/strided_slice:output:0&fruit_tn1/loop_body/Greater/y:output:0*
T0*
_output_shapes
: `
fruit_tn1/loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : »
fruit_tn1/loop_body/SelectV2SelectV2fruit_tn1/loop_body/Greater:z:03fruit_tn1/loop_body/PlaceholderWithDefault:output:0'fruit_tn1/loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: c
!fruit_tn1/loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : о
fruit_tn1/loop_body/GatherV2GatherV2 max_pooling2d_1/MaxPool:output:0%fruit_tn1/loop_body/SelectV2:output:0*fruit_tn1/loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:АТ
"fruit_tn1/loop_body/ReadVariableOpReadVariableOp+fruit_tn1_loop_body_readvariableop_resource*"
_output_shapes
:*
dtype0Ы
$fruit_tn1/loop_body/ReadVariableOp_1ReadVariableOp-fruit_tn1_loop_body_readvariableop_1_resource*'
_output_shapes
:А*
dtype0Ц
$fruit_tn1/loop_body/ReadVariableOp_2ReadVariableOp-fruit_tn1_loop_body_readvariableop_2_resource*"
_output_shapes
:*
dtype0Б
,fruit_tn1/loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ј
'fruit_tn1/loop_body/Tensordot/transpose	Transpose%fruit_tn1/loop_body/GatherV2:output:05fruit_tn1/loop_body/Tensordot/transpose/perm:output:0*
T0*#
_output_shapes
:А|
+fruit_tn1/loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      љ
%fruit_tn1/loop_body/Tensordot/ReshapeReshape+fruit_tn1/loop_body/Tensordot/transpose:y:04fruit_tn1/loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	АГ
.fruit_tn1/loop_body/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          »
)fruit_tn1/loop_body/Tensordot/transpose_1	Transpose*fruit_tn1/loop_body/ReadVariableOp:value:07fruit_tn1/loop_body/Tensordot/transpose_1/perm:output:0*
T0*"
_output_shapes
:~
-fruit_tn1/loop_body/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ¬
'fruit_tn1/loop_body/Tensordot/Reshape_1Reshape-fruit_tn1/loop_body/Tensordot/transpose_1:y:06fruit_tn1/loop_body/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:Ї
$fruit_tn1/loop_body/Tensordot/MatMulMatMul.fruit_tn1/loop_body/Tensordot/Reshape:output:00fruit_tn1/loop_body/Tensordot/Reshape_1:output:0*
T0*
_output_shapes
:	А|
#fruit_tn1/loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            Є
fruit_tn1/loop_body/TensordotReshape.fruit_tn1/loop_body/Tensordot/MatMul:product:0,fruit_tn1/loop_body/Tensordot/shape:output:0*
T0*'
_output_shapes
:АГ
.fruit_tn1/loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"           
)fruit_tn1/loop_body/Tensordot_1/transpose	Transpose,fruit_tn1/loop_body/ReadVariableOp_2:value:07fruit_tn1/loop_body/Tensordot_1/transpose/perm:output:0*
T0*"
_output_shapes
:~
-fruit_tn1/loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ¬
'fruit_tn1/loop_body/Tensordot_1/ReshapeReshape-fruit_tn1/loop_body/Tensordot_1/transpose:y:06fruit_tn1/loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:А
/fruit_tn1/loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ј
)fruit_tn1/loop_body/Tensordot_1/Reshape_1Reshape&fruit_tn1/loop_body/Tensordot:output:08fruit_tn1/loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А(ј
&fruit_tn1/loop_body/Tensordot_1/MatMulMatMul0fruit_tn1/loop_body/Tensordot_1/Reshape:output:02fruit_tn1/loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	А(В
%fruit_tn1/loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"               ¬
fruit_tn1/loop_body/Tensordot_1Reshape0fruit_tn1/loop_body/Tensordot_1/MatMul:product:0.fruit_tn1/loop_body/Tensordot_1/shape:output:0*
T0*+
_output_shapes
:А~
-fruit_tn1/loop_body/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ¬
'fruit_tn1/loop_body/Tensordot_2/ReshapeReshape,fruit_tn1/loop_body/ReadVariableOp_1:value:06fruit_tn1/loop_body/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes
:	А2Л
.fruit_tn1/loop_body/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ѕ
)fruit_tn1/loop_body/Tensordot_2/transpose	Transpose(fruit_tn1/loop_body/Tensordot_1:output:07fruit_tn1/loop_body/Tensordot_2/transpose/perm:output:0*
T0*+
_output_shapes
:АА
/fruit_tn1/loop_body/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      «
)fruit_tn1/loop_body/Tensordot_2/Reshape_1Reshape-fruit_tn1/loop_body/Tensordot_2/transpose:y:08fruit_tn1/loop_body/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А2њ
&fruit_tn1/loop_body/Tensordot_2/MatMulMatMul0fruit_tn1/loop_body/Tensordot_2/Reshape:output:02fruit_tn1/loop_body/Tensordot_2/Reshape_1:output:0*
T0*
_output_shapes

:z
%fruit_tn1/loop_body/Tensordot_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         є
fruit_tn1/loop_body/Tensordot_2Reshape0fruit_tn1/loop_body/Tensordot_2/MatMul:product:0.fruit_tn1/loop_body/Tensordot_2/shape:output:0*
T0*"
_output_shapes
:w
"fruit_tn1/loop_body/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ѓ
fruit_tn1/loop_body/transpose	Transpose(fruit_tn1/loop_body/Tensordot_2:output:0+fruit_tn1/loop_body/transpose/perm:output:0*
T0*"
_output_shapes
:Ъ
&fruit_tn1/loop_body/add/ReadVariableOpReadVariableOp/fruit_tn1_loop_body_add_readvariableop_resource*"
_output_shapes
:*
dtype0†
fruit_tn1/loop_body/addAddV2!fruit_tn1/loop_body/transpose:y:0.fruit_tn1/loop_body/add/ReadVariableOp:value:0*
T0*"
_output_shapes
:f
fruit_tn1/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Е
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
value	B :§
fruit_tn1/pfor/rangeRange#fruit_tn1/pfor/range/start:output:0fruit_tn1/Max:output:0#fruit_tn1/pfor/range/delta:output:0*#
_output_shapes
:€€€€€€€€€h
&fruit_tn1/loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : i
'fruit_tn1/loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :≤
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
value	B :Є
'fruit_tn1/loop_body/SelectV2/pfor/add_1AddV21fruit_tn1/loop_body/SelectV2/pfor/Rank_2:output:02fruit_tn1/loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: ≥
)fruit_tn1/loop_body/SelectV2/pfor/MaximumMaximum1fruit_tn1/loop_body/SelectV2/pfor/Rank_1:output:0)fruit_tn1/loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: ≥
+fruit_tn1/loop_body/SelectV2/pfor/Maximum_1Maximum+fruit_tn1/loop_body/SelectV2/pfor/add_1:z:0-fruit_tn1/loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: t
'fruit_tn1/loop_body/SelectV2/pfor/ShapeShapefruit_tn1/pfor/range:output:0*
T0*
_output_shapes
:±
%fruit_tn1/loop_body/SelectV2/pfor/subSub/fruit_tn1/loop_body/SelectV2/pfor/Maximum_1:z:01fruit_tn1/loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: y
/fruit_tn1/loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Њ
)fruit_tn1/loop_body/SelectV2/pfor/ReshapeReshape)fruit_tn1/loop_body/SelectV2/pfor/sub:z:08fruit_tn1/loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:v
,fruit_tn1/loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Љ
&fruit_tn1/loop_body/SelectV2/pfor/TileTile5fruit_tn1/loop_body/SelectV2/pfor/Tile/input:output:02fruit_tn1/loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: 
5fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
/fruit_tn1/loop_body/SelectV2/pfor/strided_sliceStridedSlice0fruit_tn1/loop_body/SelectV2/pfor/Shape:output:0>fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack:output:0@fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_1:output:0@fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskБ
7fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Г
9fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:э
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
value	B : Ѕ
(fruit_tn1/loop_body/SelectV2/pfor/concatConcatV28fruit_tn1/loop_body/SelectV2/pfor/strided_slice:output:0/fruit_tn1/loop_body/SelectV2/pfor/Tile:output:0:fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1:output:06fruit_tn1/loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ґ
+fruit_tn1/loop_body/SelectV2/pfor/Reshape_1Reshapefruit_tn1/pfor/range:output:01fruit_tn1/loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:€€€€€€€€€д
*fruit_tn1/loop_body/SelectV2/pfor/SelectV2SelectV2fruit_tn1/loop_body/Greater:z:04fruit_tn1/loop_body/SelectV2/pfor/Reshape_1:output:0'fruit_tn1/loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:€€€€€€€€€q
/fruit_tn1/loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : •
*fruit_tn1/loop_body/GatherV2/pfor/GatherV2GatherV2 max_pooling2d_1/MaxPool:output:03fruit_tn1/loop_body/SelectV2/pfor/SelectV2:output:08fruit_tn1/loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*0
_output_shapes
:€€€€€€€€€Аt
2fruit_tn1/loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :“
0fruit_tn1/loop_body/Tensordot/transpose/pfor/addAddV25fruit_tn1/loop_body/Tensordot/transpose/perm:output:0;fruit_tn1/loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:Ж
<fruit_tn1/loop_body/Tensordot/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: z
8fruit_tn1/loop_body/Tensordot/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ≠
3fruit_tn1/loop_body/Tensordot/transpose/pfor/concatConcatV2Efruit_tn1/loop_body/Tensordot/transpose/pfor/concat/values_0:output:04fruit_tn1/loop_body/Tensordot/transpose/pfor/add:z:0Afruit_tn1/loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:с
6fruit_tn1/loop_body/Tensordot/transpose/pfor/Transpose	Transpose3fruit_tn1/loop_body/GatherV2/pfor/GatherV2:output:0<fruit_tn1/loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аx
6fruit_tn1/loop_body/Tensordot/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Г
1fruit_tn1/loop_body/Tensordot/Reshape/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:04fruit_tn1/loop_body/Tensordot/Reshape/shape:output:0?fruit_tn1/loop_body/Tensordot/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:м
2fruit_tn1/loop_body/Tensordot/Reshape/pfor/ReshapeReshape:fruit_tn1/loop_body/Tensordot/transpose/pfor/Transpose:y:0:fruit_tn1/loop_body/Tensordot/Reshape/pfor/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€АЪ
/fruit_tn1/loop_body/Tensordot/MatMul/pfor/ShapeShape;fruit_tn1/loop_body/Tensordot/Reshape/pfor/Reshape:output:0*
T0*
_output_shapes
:{
9fruit_tn1/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ш
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
valueB џ
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
valueB я
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
valueB я
3fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2Reshape8fruit_tn1/loop_body/Tensordot/MatMul/pfor/split:output:2Dfruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ѕ
-fruit_tn1/loop_body/Tensordot/MatMul/pfor/mulMul:fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape:output:0<fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: а
9fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack1fruit_tn1/loop_body/Tensordot/MatMul/pfor/mul:z:0<fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:ъ
3fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape;fruit_tn1/loop_body/Tensordot/Reshape/pfor/Reshape:output:0Bfruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€№
0fruit_tn1/loop_body/Tensordot/MatMul/pfor/MatMulMatMul<fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3:output:00fruit_tn1/loop_body/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
;fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ѓ
9fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePack:fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape:output:0<fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0Dfruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:х
3fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4Reshape:fruit_tn1/loop_body/Tensordot/MatMul/pfor/MatMul:product:0Bfruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€Аp
.fruit_tn1/loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : л
)fruit_tn1/loop_body/Tensordot/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:0,fruit_tn1/loop_body/Tensordot/shape:output:07fruit_tn1/loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ж
*fruit_tn1/loop_body/Tensordot/pfor/ReshapeReshape<fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4:output:02fruit_tn1/loop_body/Tensordot/pfor/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€А|
:fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : П
5fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:08fruit_tn1/loop_body/Tensordot_1/Reshape_1/shape:output:0Cfruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:н
6fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape3fruit_tn1/loop_body/Tensordot/pfor/Reshape:output:0>fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А(†
1fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/ShapeShape?fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:Й
?fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:≠
9fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSlice:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Hfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSlice:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskй
3fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/MinimumMinimumBfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0Dfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: u
3fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :“
1fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/EqualEqual7fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Minimum:z:0<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: Й
4fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          Й
4fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
2fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/SelectSelect5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Equal:z:0=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/t:output:0=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:Л
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSlice:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: Н
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSlice:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSlice:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskє
1fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/stackPackDfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0Dfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0Dfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:И
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose	Transpose?fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€м
3fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/ReshapeReshape9fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose:y:0:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:€€€€€€€€€А(Я
3fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape_1Shape<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:}
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : А
1fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/splitSplitDfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split/split_dim:output:0<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split~
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1Reshape:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split:output:0Ffruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2Reshape:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split:output:1Ffruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3Reshape:fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split:output:2Ffruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: „
/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/mulMul>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ж
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePack>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:03fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:€
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4Reshape<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0Dfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€а
2fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul0fruit_tn1/loop_body/Tensordot_1/Reshape:output:0>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€И
=fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€є
;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePackFfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:0>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:М
5fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5Reshape<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0Dfruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€С
<fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          В
7fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1	Transpose>fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0Efruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€А(r
0fruit_tn1/loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : с
+fruit_tn1/loop_body/Tensordot_1/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:0.fruit_tn1/loop_body/Tensordot_1/shape:output:09fruit_tn1/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:н
,fruit_tn1/loop_body/Tensordot_1/pfor/ReshapeReshape;fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:04fruit_tn1/loop_body/Tensordot_1/pfor/concat:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€Аv
4fruit_tn1/loop_body/Tensordot_2/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ў
2fruit_tn1/loop_body/Tensordot_2/transpose/pfor/addAddV27fruit_tn1/loop_body/Tensordot_2/transpose/perm:output:0=fruit_tn1/loop_body/Tensordot_2/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:И
>fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: |
:fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : µ
5fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concatConcatV2Gfruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/values_0:output:06fruit_tn1/loop_body/Tensordot_2/transpose/pfor/add:z:0Cfruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:€
8fruit_tn1/loop_body/Tensordot_2/transpose/pfor/Transpose	Transpose5fruit_tn1/loop_body/Tensordot_1/pfor/Reshape:output:0>fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€А|
:fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : П
5fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:08fruit_tn1/loop_body/Tensordot_2/Reshape_1/shape:output:0Cfruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ц
6fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/ReshapeReshape<fruit_tn1/loop_body/Tensordot_2/transpose/pfor/Transpose:y:0>fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2†
1fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/ShapeShape?fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:Й
?fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:≠
9fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_sliceStridedSlice:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Hfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1StridedSlice:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskй
3fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/MinimumMinimumBfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice:output:0Dfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: u
3fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :“
1fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/EqualEqual7fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Minimum:z:0<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: Й
4fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          Й
4fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
2fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/SelectSelect5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Equal:z:0=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/t:output:0=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:Л
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2StridedSlice:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: Н
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3StridedSlice:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4StridedSlice:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1:output:0Lfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskє
1fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/stackPackDfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2:output:0Dfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3:output:0Dfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:И
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose	Transpose?fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€м
3fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/ReshapeReshape9fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose:y:0:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:А2€€€€€€€€€Я
3fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape_1Shape<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:}
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : А
1fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/splitSplitDfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split/split_dim:output:0<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split~
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1Reshape:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split:output:0Ffruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2Reshape:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split:output:1Ffruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3Reshape:fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split:output:2Ffruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: „
/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/mulMul>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ж
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shapePack>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1:output:03fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:€
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4Reshape<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0Dfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€а
2fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/MatMulMatMul0fruit_tn1/loop_body/Tensordot_2/Reshape:output:0>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€И
=fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€є
;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shapePackFfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0:output:0>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:М
5fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5Reshape<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/MatMul:product:0Dfruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€С
<fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Б
7fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1	Transpose>fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5:output:0Efruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€r
0fruit_tn1/loop_body/Tensordot_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : с
+fruit_tn1/loop_body/Tensordot_2/pfor/concatConcatV2fruit_tn1/pfor/Reshape:output:0.fruit_tn1/loop_body/Tensordot_2/shape:output:09fruit_tn1/loop_body/Tensordot_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:д
,fruit_tn1/loop_body/Tensordot_2/pfor/ReshapeReshape;fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1:y:04fruit_tn1/loop_body/Tensordot_2/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€j
(fruit_tn1/loop_body/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :і
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
value	B : Е
)fruit_tn1/loop_body/transpose/pfor/concatConcatV2;fruit_tn1/loop_body/transpose/pfor/concat/values_0:output:0*fruit_tn1/loop_body/transpose/pfor/add:z:07fruit_tn1/loop_body/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ё
,fruit_tn1/loop_body/transpose/pfor/Transpose	Transpose5fruit_tn1/loop_body/Tensordot_2/pfor/Reshape:output:02fruit_tn1/loop_body/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€c
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
value	B :•
 fruit_tn1/loop_body/add/pfor/addAddV2,fruit_tn1/loop_body/add/pfor/Rank_1:output:0+fruit_tn1/loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: Ґ
$fruit_tn1/loop_body/add/pfor/MaximumMaximum$fruit_tn1/loop_body/add/pfor/add:z:0*fruit_tn1/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: В
"fruit_tn1/loop_body/add/pfor/ShapeShape0fruit_tn1/loop_body/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:Ю
 fruit_tn1/loop_body/add/pfor/subSub(fruit_tn1/loop_body/add/pfor/Maximum:z:0*fruit_tn1/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: t
*fruit_tn1/loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:ѓ
$fruit_tn1/loop_body/add/pfor/ReshapeReshape$fruit_tn1/loop_body/add/pfor/sub:z:03fruit_tn1/loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:q
'fruit_tn1/loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:≠
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
valueB:а
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
valueB:ж
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
value	B : ®
#fruit_tn1/loop_body/add/pfor/concatConcatV23fruit_tn1/loop_body/add/pfor/strided_slice:output:0*fruit_tn1/loop_body/add/pfor/Tile:output:05fruit_tn1/loop_body/add/pfor/strided_slice_1:output:01fruit_tn1/loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ћ
&fruit_tn1/loop_body/add/pfor/Reshape_1Reshape0fruit_tn1/loop_body/transpose/pfor/Transpose:y:0,fruit_tn1/loop_body/add/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€∆
"fruit_tn1/loop_body/add/pfor/AddV2AddV2/fruit_tn1/loop_body/add/pfor/Reshape_1:output:0.fruit_tn1/loop_body/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€h
fruit_tn1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   Ш
fruit_tn1/ReshapeReshape&fruit_tn1/loop_body/add/pfor/AddV2:z:0 fruit_tn1/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@d
fruit_tn1/ReluRelufruit_tn1/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@~
$fruit_tn1/ActivityRegularizer/SquareSquarefruit_tn1/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@t
#fruit_tn1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       °
!fruit_tn1/ActivityRegularizer/SumSum(fruit_tn1/ActivityRegularizer/Square:y:0,fruit_tn1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: h
#fruit_tn1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ђ≈'7£
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
valueB:з
+fruit_tn1/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn1/ActivityRegularizer/Shape:output:0:fruit_tn1/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
"fruit_tn1/ActivityRegularizer/CastCast4fruit_tn1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: †
%fruit_tn1/ActivityRegularizer/truedivRealDiv%fruit_tn1/ActivityRegularizer/mul:z:0&fruit_tn1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: m
dropout1/IdentityIdentityfruit_tn1/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@Y
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
valueB:Г
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
value	B :Н
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
value	B : ±
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
valueB:µ
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
value	B :Ы
fruit_tn2/loop_body/GreaterGreater*fruit_tn2/loop_body/strided_slice:output:0&fruit_tn2/loop_body/Greater/y:output:0*
T0*
_output_shapes
: `
fruit_tn2/loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : »
fruit_tn2/loop_body/SelectV2SelectV2fruit_tn2/loop_body/Greater:z:03fruit_tn2/loop_body/PlaceholderWithDefault:output:0'fruit_tn2/loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: c
!fruit_tn2/loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : я
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
valueB"         ¶
fruit_tn2/loop_body/ReshapeReshape%fruit_tn2/loop_body/GatherV2:output:0*fruit_tn2/loop_body/Reshape/shape:output:0*
T0*"
_output_shapes
:О
"fruit_tn2/loop_body/ReadVariableOpReadVariableOp+fruit_tn2_loop_body_readvariableop_resource*
_output_shapes

:*
dtype0Ы
$fruit_tn2/loop_body/ReadVariableOp_1ReadVariableOp-fruit_tn2_loop_body_readvariableop_1_resource*'
_output_shapes
:Г*
dtype0Т
$fruit_tn2/loop_body/ReadVariableOp_2ReadVariableOp-fruit_tn2_loop_body_readvariableop_2_resource*
_output_shapes

:*
dtype0Б
,fruit_tn2/loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Њ
'fruit_tn2/loop_body/Tensordot/transpose	Transpose$fruit_tn2/loop_body/Reshape:output:05fruit_tn2/loop_body/Tensordot/transpose/perm:output:0*
T0*"
_output_shapes
:|
+fruit_tn2/loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Љ
%fruit_tn2/loop_body/Tensordot/ReshapeReshape+fruit_tn2/loop_body/Tensordot/transpose:y:04fruit_tn2/loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:≥
$fruit_tn2/loop_body/Tensordot/MatMulMatMul.fruit_tn2/loop_body/Tensordot/Reshape:output:0*fruit_tn2/loop_body/ReadVariableOp:value:0*
T0*
_output_shapes

:x
#fruit_tn2/loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ≥
fruit_tn2/loop_body/TensordotReshape.fruit_tn2/loop_body/Tensordot/MatMul:product:0,fruit_tn2/loop_body/Tensordot/shape:output:0*
T0*"
_output_shapes
:З
.fruit_tn2/loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ѕ
)fruit_tn2/loop_body/Tensordot_1/transpose	Transpose,fruit_tn2/loop_body/ReadVariableOp_1:value:07fruit_tn2/loop_body/Tensordot_1/transpose/perm:output:0*
T0*'
_output_shapes
:Г~
-fruit_tn2/loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"     √
'fruit_tn2/loop_body/Tensordot_1/ReshapeReshape-fruit_tn2/loop_body/Tensordot_1/transpose:y:06fruit_tn2/loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes
:	МЕ
0fruit_tn2/loop_body/Tensordot_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          »
+fruit_tn2/loop_body/Tensordot_1/transpose_1	Transpose&fruit_tn2/loop_body/Tensordot:output:09fruit_tn2/loop_body/Tensordot_1/transpose_1/perm:output:0*
T0*"
_output_shapes
:А
/fruit_tn2/loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      »
)fruit_tn2/loop_body/Tensordot_1/Reshape_1Reshape/fruit_tn2/loop_body/Tensordot_1/transpose_1:y:08fruit_tn2/loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:ј
&fruit_tn2/loop_body/Tensordot_1/MatMulMatMul0fruit_tn2/loop_body/Tensordot_1/Reshape:output:02fruit_tn2/loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	Мz
%fruit_tn2/loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"Г         Ї
fruit_tn2/loop_body/Tensordot_1Reshape0fruit_tn2/loop_body/Tensordot_1/MatMul:product:0.fruit_tn2/loop_body/Tensordot_1/shape:output:0*
T0*#
_output_shapes
:Г~
-fruit_tn2/loop_body/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ѕ
'fruit_tn2/loop_body/Tensordot_2/ReshapeReshape,fruit_tn2/loop_body/ReadVariableOp_2:value:06fruit_tn2/loop_body/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes

:Г
.fruit_tn2/loop_body/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
)fruit_tn2/loop_body/Tensordot_2/transpose	Transpose(fruit_tn2/loop_body/Tensordot_1:output:07fruit_tn2/loop_body/Tensordot_2/transpose/perm:output:0*
T0*#
_output_shapes
:ГА
/fruit_tn2/loop_body/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   Г   «
)fruit_tn2/loop_body/Tensordot_2/Reshape_1Reshape-fruit_tn2/loop_body/Tensordot_2/transpose:y:08fruit_tn2/loop_body/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	Гј
&fruit_tn2/loop_body/Tensordot_2/MatMulMatMul0fruit_tn2/loop_body/Tensordot_2/Reshape:output:02fruit_tn2/loop_body/Tensordot_2/Reshape_1:output:0*
T0*
_output_shapes
:	Гp
%fruit_tn2/loop_body/Tensordot_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:Г≤
fruit_tn2/loop_body/Tensordot_2Reshape0fruit_tn2/loop_body/Tensordot_2/MatMul:product:0.fruit_tn2/loop_body/Tensordot_2/shape:output:0*
T0*
_output_shapes	
:ГУ
&fruit_tn2/loop_body/add/ReadVariableOpReadVariableOp/fruit_tn2_loop_body_add_readvariableop_resource*
_output_shapes	
:Г*
dtype0†
fruit_tn2/loop_body/addAddV2(fruit_tn2/loop_body/Tensordot_2:output:0.fruit_tn2/loop_body/add/ReadVariableOp:value:0*
T0*
_output_shapes	
:Гf
fruit_tn2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Е
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
value	B :§
fruit_tn2/pfor/rangeRange#fruit_tn2/pfor/range/start:output:0fruit_tn2/Max:output:0#fruit_tn2/pfor/range/delta:output:0*#
_output_shapes
:€€€€€€€€€h
&fruit_tn2/loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : i
'fruit_tn2/loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :≤
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
value	B :Є
'fruit_tn2/loop_body/SelectV2/pfor/add_1AddV21fruit_tn2/loop_body/SelectV2/pfor/Rank_2:output:02fruit_tn2/loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: ≥
)fruit_tn2/loop_body/SelectV2/pfor/MaximumMaximum1fruit_tn2/loop_body/SelectV2/pfor/Rank_1:output:0)fruit_tn2/loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: ≥
+fruit_tn2/loop_body/SelectV2/pfor/Maximum_1Maximum+fruit_tn2/loop_body/SelectV2/pfor/add_1:z:0-fruit_tn2/loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: t
'fruit_tn2/loop_body/SelectV2/pfor/ShapeShapefruit_tn2/pfor/range:output:0*
T0*
_output_shapes
:±
%fruit_tn2/loop_body/SelectV2/pfor/subSub/fruit_tn2/loop_body/SelectV2/pfor/Maximum_1:z:01fruit_tn2/loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: y
/fruit_tn2/loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Њ
)fruit_tn2/loop_body/SelectV2/pfor/ReshapeReshape)fruit_tn2/loop_body/SelectV2/pfor/sub:z:08fruit_tn2/loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:v
,fruit_tn2/loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Љ
&fruit_tn2/loop_body/SelectV2/pfor/TileTile5fruit_tn2/loop_body/SelectV2/pfor/Tile/input:output:02fruit_tn2/loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: 
5fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
/fruit_tn2/loop_body/SelectV2/pfor/strided_sliceStridedSlice0fruit_tn2/loop_body/SelectV2/pfor/Shape:output:0>fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack:output:0@fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_1:output:0@fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskБ
7fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Г
9fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:э
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
value	B : Ѕ
(fruit_tn2/loop_body/SelectV2/pfor/concatConcatV28fruit_tn2/loop_body/SelectV2/pfor/strided_slice:output:0/fruit_tn2/loop_body/SelectV2/pfor/Tile:output:0:fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1:output:06fruit_tn2/loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ґ
+fruit_tn2/loop_body/SelectV2/pfor/Reshape_1Reshapefruit_tn2/pfor/range:output:01fruit_tn2/loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:€€€€€€€€€д
*fruit_tn2/loop_body/SelectV2/pfor/SelectV2SelectV2fruit_tn2/loop_body/Greater:z:04fruit_tn2/loop_body/SelectV2/pfor/Reshape_1:output:0'fruit_tn2/loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:€€€€€€€€€q
/fruit_tn2/loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ц
*fruit_tn2/loop_body/GatherV2/pfor/GatherV2GatherV2dropout1/Identity:output:03fruit_tn2/loop_body/SelectV2/pfor/SelectV2:output:08fruit_tn2/loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:€€€€€€€€€@n
,fruit_tn2/loop_body/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : е
'fruit_tn2/loop_body/Reshape/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:0*fruit_tn2/loop_body/Reshape/shape:output:05fruit_tn2/loop_body/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:‘
(fruit_tn2/loop_body/Reshape/pfor/ReshapeReshape3fruit_tn2/loop_body/GatherV2/pfor/GatherV2:output:00fruit_tn2/loop_body/Reshape/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€t
2fruit_tn2/loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :“
0fruit_tn2/loop_body/Tensordot/transpose/pfor/addAddV25fruit_tn2/loop_body/Tensordot/transpose/perm:output:0;fruit_tn2/loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:Ж
<fruit_tn2/loop_body/Tensordot/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: z
8fruit_tn2/loop_body/Tensordot/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ≠
3fruit_tn2/loop_body/Tensordot/transpose/pfor/concatConcatV2Efruit_tn2/loop_body/Tensordot/transpose/pfor/concat/values_0:output:04fruit_tn2/loop_body/Tensordot/transpose/pfor/add:z:0Afruit_tn2/loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:о
6fruit_tn2/loop_body/Tensordot/transpose/pfor/Transpose	Transpose1fruit_tn2/loop_body/Reshape/pfor/Reshape:output:0<fruit_tn2/loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€x
6fruit_tn2/loop_body/Tensordot/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Г
1fruit_tn2/loop_body/Tensordot/Reshape/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:04fruit_tn2/loop_body/Tensordot/Reshape/shape:output:0?fruit_tn2/loop_body/Tensordot/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:л
2fruit_tn2/loop_body/Tensordot/Reshape/pfor/ReshapeReshape:fruit_tn2/loop_body/Tensordot/transpose/pfor/Transpose:y:0:fruit_tn2/loop_body/Tensordot/Reshape/pfor/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€Ъ
/fruit_tn2/loop_body/Tensordot/MatMul/pfor/ShapeShape;fruit_tn2/loop_body/Tensordot/Reshape/pfor/Reshape:output:0*
T0*
_output_shapes
:{
9fruit_tn2/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ш
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
valueB џ
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
valueB я
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
valueB я
3fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2Reshape8fruit_tn2/loop_body/Tensordot/MatMul/pfor/split:output:2Dfruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ѕ
-fruit_tn2/loop_body/Tensordot/MatMul/pfor/mulMul:fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape:output:0<fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: а
9fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack1fruit_tn2/loop_body/Tensordot/MatMul/pfor/mul:z:0<fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:ъ
3fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape;fruit_tn2/loop_body/Tensordot/Reshape/pfor/Reshape:output:0Bfruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€÷
0fruit_tn2/loop_body/Tensordot/MatMul/pfor/MatMulMatMul<fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0*fruit_tn2/loop_body/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
;fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ѓ
9fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePack:fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape:output:0<fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0Dfruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:ф
3fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4Reshape:fruit_tn2/loop_body/Tensordot/MatMul/pfor/MatMul:product:0Bfruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€p
.fruit_tn2/loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : л
)fruit_tn2/loop_body/Tensordot/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:0,fruit_tn2/loop_body/Tensordot/shape:output:07fruit_tn2/loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:б
*fruit_tn2/loop_body/Tensordot/pfor/ReshapeReshape<fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4:output:02fruit_tn2/loop_body/Tensordot/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€x
6fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :ё
4fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/addAddV29fruit_tn2/loop_body/Tensordot_1/transpose_1/perm:output:0?fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/add/y:output:0*
T0*
_output_shapes
:К
@fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: ~
<fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : љ
7fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concatConcatV2Ifruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/values_0:output:08fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/add:z:0Efruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ш
:fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/Transpose	Transpose3fruit_tn2/loop_body/Tensordot/pfor/Reshape:output:0@fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€|
:fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : П
5fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:08fruit_tn2/loop_body/Tensordot_1/Reshape_1/shape:output:0Cfruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ч
6fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape>fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/Transpose:y:0>fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€†
1fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/ShapeShape?fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:Й
?fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:≠
9fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSlice:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Hfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSlice:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskй
3fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/MinimumMinimumBfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0Dfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: u
3fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :“
1fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/EqualEqual7fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Minimum:z:0<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: Й
4fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          Й
4fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
2fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/SelectSelect5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Equal:z:0=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/t:output:0=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:Л
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSlice:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: Н
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSlice:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSlice:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskє
1fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/stackPackDfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0Dfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0Dfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:И
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose	Transpose?fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€л
3fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/ReshapeReshape9fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose:y:0:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*+
_output_shapes
:€€€€€€€€€Я
3fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape_1Shape<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:}
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : А
1fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/splitSplitDfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split/split_dim:output:0<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split~
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1Reshape:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split:output:0Ffruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2Reshape:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split:output:1Ffruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3Reshape:fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split:output:2Ffruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: „
/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/mulMul>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ж
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePack>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:03fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:€
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4Reshape<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0Dfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€б
2fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul0fruit_tn2/loop_body/Tensordot_1/Reshape:output:0>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*(
_output_shapes
:М€€€€€€€€€И
=fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€є
;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePackFfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:0>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:М
5fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5Reshape<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0Dfruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€С
<fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          В
7fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1	Transpose>fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0Efruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Мr
0fruit_tn2/loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : с
+fruit_tn2/loop_body/Tensordot_1/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:0.fruit_tn2/loop_body/Tensordot_1/shape:output:09fruit_tn2/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:е
,fruit_tn2/loop_body/Tensordot_1/pfor/ReshapeReshape;fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:04fruit_tn2/loop_body/Tensordot_1/pfor/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€Гv
4fruit_tn2/loop_body/Tensordot_2/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ў
2fruit_tn2/loop_body/Tensordot_2/transpose/pfor/addAddV27fruit_tn2/loop_body/Tensordot_2/transpose/perm:output:0=fruit_tn2/loop_body/Tensordot_2/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:И
>fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: |
:fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : µ
5fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concatConcatV2Gfruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/values_0:output:06fruit_tn2/loop_body/Tensordot_2/transpose/pfor/add:z:0Cfruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ч
8fruit_tn2/loop_body/Tensordot_2/transpose/pfor/Transpose	Transpose5fruit_tn2/loop_body/Tensordot_1/pfor/Reshape:output:0>fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€Г|
:fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : П
5fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:08fruit_tn2/loop_body/Tensordot_2/Reshape_1/shape:output:0Cfruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ц
6fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/ReshapeReshape<fruit_tn2/loop_body/Tensordot_2/transpose/pfor/Transpose:y:0>fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Г†
1fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/ShapeShape?fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:Й
?fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:≠
9fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_sliceStridedSlice:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Hfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1StridedSlice:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskй
3fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/MinimumMinimumBfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice:output:0Dfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: u
3fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :“
1fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/EqualEqual7fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Minimum:z:0<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: Й
4fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          Й
4fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
2fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/SelectSelect5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Equal:z:0=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/t:output:0=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:Л
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2StridedSlice:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: Н
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3StridedSlice:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЛ
Afruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Cfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4StridedSlice:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Jfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1:output:0Lfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskє
1fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/stackPackDfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2:output:0Dfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3:output:0Dfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:И
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose	Transpose?fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€м
3fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/ReshapeReshape9fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose:y:0:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:€€€€€€€€€ГЯ
3fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape_1Shape<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:}
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : А
1fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/splitSplitDfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split/split_dim:output:0<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_split~
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1Reshape:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split:output:0Ffruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2Reshape:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split:output:1Ffruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ~
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB А
=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB е
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3Reshape:fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split:output:2Ffruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: „
/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/mulMul>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: ж
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shapePack>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1:output:03fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:€
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4Reshape<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0Dfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€а
2fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/MatMulMatMul0fruit_tn2/loop_body/Tensordot_2/Reshape:output:0>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€И
=fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€є
;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shapePackFfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0:output:0>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:М
5fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5Reshape<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/MatMul:product:0Dfruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€С
<fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          В
7fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1	Transpose>fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5:output:0Efruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Гr
0fruit_tn2/loop_body/Tensordot_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : с
+fruit_tn2/loop_body/Tensordot_2/pfor/concatConcatV2fruit_tn2/pfor/Reshape:output:0.fruit_tn2/loop_body/Tensordot_2/shape:output:09fruit_tn2/loop_body/Tensordot_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ё
,fruit_tn2/loop_body/Tensordot_2/pfor/ReshapeReshape;fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1:y:04fruit_tn2/loop_body/Tensordot_2/pfor/concat:output:0*
T0*(
_output_shapes
:€€€€€€€€€Гc
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
value	B :•
 fruit_tn2/loop_body/add/pfor/addAddV2,fruit_tn2/loop_body/add/pfor/Rank_1:output:0+fruit_tn2/loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: Ґ
$fruit_tn2/loop_body/add/pfor/MaximumMaximum$fruit_tn2/loop_body/add/pfor/add:z:0*fruit_tn2/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: З
"fruit_tn2/loop_body/add/pfor/ShapeShape5fruit_tn2/loop_body/Tensordot_2/pfor/Reshape:output:0*
T0*
_output_shapes
:Ю
 fruit_tn2/loop_body/add/pfor/subSub(fruit_tn2/loop_body/add/pfor/Maximum:z:0*fruit_tn2/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: t
*fruit_tn2/loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:ѓ
$fruit_tn2/loop_body/add/pfor/ReshapeReshape$fruit_tn2/loop_body/add/pfor/sub:z:03fruit_tn2/loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:q
'fruit_tn2/loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:≠
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
valueB:а
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
valueB:ж
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
value	B : ®
#fruit_tn2/loop_body/add/pfor/concatConcatV23fruit_tn2/loop_body/add/pfor/strided_slice:output:0*fruit_tn2/loop_body/add/pfor/Tile:output:05fruit_tn2/loop_body/add/pfor/strided_slice_1:output:01fruit_tn2/loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:…
&fruit_tn2/loop_body/add/pfor/Reshape_1Reshape5fruit_tn2/loop_body/Tensordot_2/pfor/Reshape:output:0,fruit_tn2/loop_body/add/pfor/concat:output:0*
T0*(
_output_shapes
:€€€€€€€€€Гњ
"fruit_tn2/loop_body/add/pfor/AddV2AddV2/fruit_tn2/loop_body/add/pfor/Reshape_1:output:0.fruit_tn2/loop_body/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Гh
fruit_tn2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Г   Щ
fruit_tn2/ReshapeReshape&fruit_tn2/loop_body/add/pfor/AddV2:z:0 fruit_tn2/Reshape/shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€Гk
fruit_tn2/SoftmaxSoftmaxfruit_tn2/Reshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€Г~
$fruit_tn2/ActivityRegularizer/SquareSquarefruit_tn2/Softmax:softmax:0*
T0*(
_output_shapes
:€€€€€€€€€Гt
#fruit_tn2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       °
!fruit_tn2/ActivityRegularizer/SumSum(fruit_tn2/ActivityRegularizer/Square:y:0,fruit_tn2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: h
#fruit_tn2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ђ≈'7£
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
valueB:з
+fruit_tn2/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn2/ActivityRegularizer/Shape:output:0:fruit_tn2/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
"fruit_tn2/ActivityRegularizer/CastCast4fruit_tn2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: †
%fruit_tn2/ActivityRegularizer/truedivRealDiv%fruit_tn2/ActivityRegularizer/mul:z:0&fruit_tn2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: k
IdentityIdentityfruit_tn2/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Гi

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
: ї
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
?:€€€€€€€€€22: : : : : : : : : : : : : : : : : : 2>
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
:€€€€€€€€€22
 
_user_specified_nameinputs
Р√	
Ь
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
:sequential_1_conv2dmpo_1_reshape_2_readvariableop_resource:	АN
8sequential_1_fruit_tn1_loop_body_readvariableop_resource:U
:sequential_1_fruit_tn1_loop_body_readvariableop_1_resource:АP
:sequential_1_fruit_tn1_loop_body_readvariableop_2_resource:R
<sequential_1_fruit_tn1_loop_body_add_readvariableop_resource:J
8sequential_1_fruit_tn2_loop_body_readvariableop_resource:U
:sequential_1_fruit_tn2_loop_body_readvariableop_1_resource:ГL
:sequential_1_fruit_tn2_loop_body_readvariableop_2_resource:K
<sequential_1_fruit_tn2_loop_body_add_readvariableop_resource:	Г
identityИҐ*sequential_1/conv2d/BiasAdd/ReadVariableOpҐ)sequential_1/conv2d/Conv2D/ReadVariableOpҐ%sequential_1/conv2dmpo/ReadVariableOpҐ'sequential_1/conv2dmpo/ReadVariableOp_1Ґ/sequential_1/conv2dmpo/Reshape_2/ReadVariableOpҐ'sequential_1/conv2dmpo_1/ReadVariableOpҐ)sequential_1/conv2dmpo_1/ReadVariableOp_1Ґ)sequential_1/conv2dmpo_1/ReadVariableOp_2Ґ)sequential_1/conv2dmpo_1/ReadVariableOp_3Ґ1sequential_1/conv2dmpo_1/Reshape_2/ReadVariableOpҐ/sequential_1/fruit_tn1/loop_body/ReadVariableOpҐ1sequential_1/fruit_tn1/loop_body/ReadVariableOp_1Ґ1sequential_1/fruit_tn1/loop_body/ReadVariableOp_2Ґ3sequential_1/fruit_tn1/loop_body/add/ReadVariableOpҐ/sequential_1/fruit_tn2/loop_body/ReadVariableOpҐ1sequential_1/fruit_tn2/loop_body/ReadVariableOp_1Ґ1sequential_1/fruit_tn2/loop_body/ReadVariableOp_2Ґ3sequential_1/fruit_tn2/loop_body/add/ReadVariableOp§
)sequential_1/conv2d/Conv2D/ReadVariableOpReadVariableOp2sequential_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ƒ
sequential_1/conv2d/Conv2DConv2Dinput_121sequential_1/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€//*
paddingVALID*
strides
Ъ
*sequential_1/conv2d/BiasAdd/ReadVariableOpReadVariableOp3sequential_1_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0є
sequential_1/conv2d/BiasAddBiasAdd#sequential_1/conv2d/Conv2D:output:02sequential_1/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€//Ь
%sequential_1/conv2dmpo/ReadVariableOpReadVariableOp.sequential_1_conv2dmpo_readvariableop_resource*&
_output_shapes
:*
dtype0†
'sequential_1/conv2dmpo/ReadVariableOp_1ReadVariableOp0sequential_1_conv2dmpo_readvariableop_1_resource*&
_output_shapes
:*
dtype0И
/sequential_1/conv2dmpo/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ”
*sequential_1/conv2dmpo/Tensordot/transpose	Transpose/sequential_1/conv2dmpo/ReadVariableOp_1:value:08sequential_1/conv2dmpo/Tensordot/transpose/perm:output:0*
T0*&
_output_shapes
:
.sequential_1/conv2dmpo/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ≈
(sequential_1/conv2dmpo/Tensordot/ReshapeReshape.sequential_1/conv2dmpo/Tensordot/transpose:y:07sequential_1/conv2dmpo/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:К
1sequential_1/conv2dmpo/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ’
,sequential_1/conv2dmpo/Tensordot/transpose_1	Transpose-sequential_1/conv2dmpo/ReadVariableOp:value:0:sequential_1/conv2dmpo/Tensordot/transpose_1/perm:output:0*
T0*&
_output_shapes
:Б
0sequential_1/conv2dmpo/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ћ
*sequential_1/conv2dmpo/Tensordot/Reshape_1Reshape0sequential_1/conv2dmpo/Tensordot/transpose_1:y:09sequential_1/conv2dmpo/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:¬
'sequential_1/conv2dmpo/Tensordot/MatMulMatMul1sequential_1/conv2dmpo/Tensordot/Reshape:output:03sequential_1/conv2dmpo/Tensordot/Reshape_1:output:0*
T0*
_output_shapes

:З
&sequential_1/conv2dmpo/Tensordot/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  »
 sequential_1/conv2dmpo/TensordotReshape1sequential_1/conv2dmpo/Tensordot/MatMul:product:0/sequential_1/conv2dmpo/Tensordot/shape:output:0*
T0*.
_output_shapes
:Ж
%sequential_1/conv2dmpo/transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   Ѕ
 sequential_1/conv2dmpo/transpose	Transpose)sequential_1/conv2dmpo/Tensordot:output:0.sequential_1/conv2dmpo/transpose/perm:output:0*
T0*.
_output_shapes
:И
'sequential_1/conv2dmpo/transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ј
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
valueB:ј
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
valueB: Ъ
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
valueB: 
&sequential_1/conv2dmpo/strided_slice_1StridedSlice%sequential_1/conv2dmpo/Shape:output:05sequential_1/conv2dmpo/strided_slice_1/stack:output:07sequential_1/conv2dmpo/strided_slice_1/stack_1:output:07sequential_1/conv2dmpo/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskВ
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
€€€€€€€€€ж
sequential_1/conv2dmpo/concatConcatV2/sequential_1/conv2dmpo/strided_slice_1:output:0/sequential_1/conv2dmpo/concat/values_1:output:0+sequential_1/conv2dmpo/concat/axis:output:0*
N*
T0*
_output_shapes
:Ѓ
sequential_1/conv2dmpo/ReshapeReshape&sequential_1/conv2dmpo/transpose_1:y:0&sequential_1/conv2dmpo/concat:output:0*
T0**
_output_shapes
:Д
'sequential_1/conv2dmpo/transpose_2/permConst*
_output_shapes
:*
dtype0*)
value B"                њ
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
valueB: 
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
valueB: †
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
valueB:ћ
&sequential_1/conv2dmpo/strided_slice_3StridedSlice'sequential_1/conv2dmpo/Shape_1:output:05sequential_1/conv2dmpo/strided_slice_3/stack:output:07sequential_1/conv2dmpo/strided_slice_3/stack_1:output:07sequential_1/conv2dmpo/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskЖ
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
€€€€€€€€€м
sequential_1/conv2dmpo/concat_1ConcatV2/sequential_1/conv2dmpo/strided_slice_3:output:01sequential_1/conv2dmpo/concat_1/values_1:output:0-sequential_1/conv2dmpo/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ѓ
 sequential_1/conv2dmpo/Reshape_1Reshape&sequential_1/conv2dmpo/transpose_2:y:0(sequential_1/conv2dmpo/concat_1:output:0*
T0*&
_output_shapes
:Џ
sequential_1/conv2dmpo/Conv2DConv2D$sequential_1/conv2d/BiasAdd:output:0)sequential_1/conv2dmpo/Reshape_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€//*
paddingSAME*
strides
§
/sequential_1/conv2dmpo/Reshape_2/ReadVariableOpReadVariableOp8sequential_1_conv2dmpo_reshape_2_readvariableop_resource*
_output_shapes
:*
dtype0w
&sequential_1/conv2dmpo/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Њ
 sequential_1/conv2dmpo/Reshape_2Reshape7sequential_1/conv2dmpo/Reshape_2/ReadVariableOp:value:0/sequential_1/conv2dmpo/Reshape_2/shape:output:0*
T0*
_output_shapes

:∞
sequential_1/conv2dmpo/addAddV2&sequential_1/conv2dmpo/Conv2D:output:0)sequential_1/conv2dmpo/Reshape_2:output:0*
T0*/
_output_shapes
:€€€€€€€€€//}
sequential_1/conv2dmpo/ReluRelusequential_1/conv2dmpo/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€//†
1sequential_1/conv2dmpo/ActivityRegularizer/SquareSquare)sequential_1/conv2dmpo/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€//Й
0sequential_1/conv2dmpo/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             »
.sequential_1/conv2dmpo/ActivityRegularizer/SumSum5sequential_1/conv2dmpo/ActivityRegularizer/Square:y:09sequential_1/conv2dmpo/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: u
0sequential_1/conv2dmpo/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ђ≈'7 
.sequential_1/conv2dmpo/ActivityRegularizer/mulMul9sequential_1/conv2dmpo/ActivityRegularizer/mul/x:output:07sequential_1/conv2dmpo/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: Й
0sequential_1/conv2dmpo/ActivityRegularizer/ShapeShape)sequential_1/conv2dmpo/Relu:activations:0*
T0*
_output_shapes
:И
>sequential_1/conv2dmpo/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: К
@sequential_1/conv2dmpo/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:К
@sequential_1/conv2dmpo/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
8sequential_1/conv2dmpo/ActivityRegularizer/strided_sliceStridedSlice9sequential_1/conv2dmpo/ActivityRegularizer/Shape:output:0Gsequential_1/conv2dmpo/ActivityRegularizer/strided_slice/stack:output:0Isequential_1/conv2dmpo/ActivityRegularizer/strided_slice/stack_1:output:0Isequential_1/conv2dmpo/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask™
/sequential_1/conv2dmpo/ActivityRegularizer/CastCastAsequential_1/conv2dmpo/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: «
2sequential_1/conv2dmpo/ActivityRegularizer/truedivRealDiv2sequential_1/conv2dmpo/ActivityRegularizer/mul:z:03sequential_1/conv2dmpo/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ≈
"sequential_1/max_pooling2d/MaxPoolMaxPool)sequential_1/conv2dmpo/Relu:activations:0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
†
'sequential_1/conv2dmpo_1/ReadVariableOpReadVariableOp0sequential_1_conv2dmpo_1_readvariableop_resource*&
_output_shapes
:*
dtype0§
)sequential_1/conv2dmpo_1/ReadVariableOp_1ReadVariableOp2sequential_1_conv2dmpo_1_readvariableop_1_resource*&
_output_shapes
:*
dtype0§
)sequential_1/conv2dmpo_1/ReadVariableOp_2ReadVariableOp2sequential_1_conv2dmpo_1_readvariableop_2_resource*&
_output_shapes
:*
dtype0§
)sequential_1/conv2dmpo_1/ReadVariableOp_3ReadVariableOp2sequential_1_conv2dmpo_1_readvariableop_3_resource*&
_output_shapes
:*
dtype0К
1sequential_1/conv2dmpo_1/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ў
,sequential_1/conv2dmpo_1/Tensordot/transpose	Transpose1sequential_1/conv2dmpo_1/ReadVariableOp_1:value:0:sequential_1/conv2dmpo_1/Tensordot/transpose/perm:output:0*
T0*&
_output_shapes
:Б
0sequential_1/conv2dmpo_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@      Ћ
*sequential_1/conv2dmpo_1/Tensordot/ReshapeReshape0sequential_1/conv2dmpo_1/Tensordot/transpose:y:09sequential_1/conv2dmpo_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@М
3sequential_1/conv2dmpo_1/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             џ
.sequential_1/conv2dmpo_1/Tensordot/transpose_1	Transpose/sequential_1/conv2dmpo_1/ReadVariableOp:value:0<sequential_1/conv2dmpo_1/Tensordot/transpose_1/perm:output:0*
T0*&
_output_shapes
:Г
2sequential_1/conv2dmpo_1/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      —
,sequential_1/conv2dmpo_1/Tensordot/Reshape_1Reshape2sequential_1/conv2dmpo_1/Tensordot/transpose_1:y:0;sequential_1/conv2dmpo_1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:»
)sequential_1/conv2dmpo_1/Tensordot/MatMulMatMul3sequential_1/conv2dmpo_1/Tensordot/Reshape:output:05sequential_1/conv2dmpo_1/Tensordot/Reshape_1:output:0*
T0*
_output_shapes

:@Й
(sequential_1/conv2dmpo_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  ќ
"sequential_1/conv2dmpo_1/TensordotReshape3sequential_1/conv2dmpo_1/Tensordot/MatMul:product:01sequential_1/conv2dmpo_1/Tensordot/shape:output:0*
T0*.
_output_shapes
:М
3sequential_1/conv2dmpo_1/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ё
.sequential_1/conv2dmpo_1/Tensordot_1/transpose	Transpose1sequential_1/conv2dmpo_1/ReadVariableOp_2:value:0<sequential_1/conv2dmpo_1/Tensordot_1/transpose/perm:output:0*
T0*&
_output_shapes
:Г
2sequential_1/conv2dmpo_1/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@      —
,sequential_1/conv2dmpo_1/Tensordot_1/ReshapeReshape2sequential_1/conv2dmpo_1/Tensordot_1/transpose:y:0;sequential_1/conv2dmpo_1/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:@О
5sequential_1/conv2dmpo_1/Tensordot_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             б
0sequential_1/conv2dmpo_1/Tensordot_1/transpose_1	Transpose1sequential_1/conv2dmpo_1/ReadVariableOp_3:value:0>sequential_1/conv2dmpo_1/Tensordot_1/transpose_1/perm:output:0*
T0*&
_output_shapes
:Е
4sequential_1/conv2dmpo_1/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      „
.sequential_1/conv2dmpo_1/Tensordot_1/Reshape_1Reshape4sequential_1/conv2dmpo_1/Tensordot_1/transpose_1:y:0=sequential_1/conv2dmpo_1/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:ќ
+sequential_1/conv2dmpo_1/Tensordot_1/MatMulMatMul5sequential_1/conv2dmpo_1/Tensordot_1/Reshape:output:07sequential_1/conv2dmpo_1/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes

:@Л
*sequential_1/conv2dmpo_1/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  ‘
$sequential_1/conv2dmpo_1/Tensordot_1Reshape5sequential_1/conv2dmpo_1/Tensordot_1/MatMul:product:03sequential_1/conv2dmpo_1/Tensordot_1/shape:output:0*
T0*.
_output_shapes
:Ф
3sequential_1/conv2dmpo_1/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   я
.sequential_1/conv2dmpo_1/Tensordot_2/transpose	Transpose+sequential_1/conv2dmpo_1/Tensordot:output:0<sequential_1/conv2dmpo_1/Tensordot_2/transpose/perm:output:0*
T0*.
_output_shapes
:Г
2sequential_1/conv2dmpo_1/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"А      “
,sequential_1/conv2dmpo_1/Tensordot_2/ReshapeReshape2sequential_1/conv2dmpo_1/Tensordot_2/transpose:y:0;sequential_1/conv2dmpo_1/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes
:	АЦ
5sequential_1/conv2dmpo_1/Tensordot_2/transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   е
0sequential_1/conv2dmpo_1/Tensordot_2/transpose_1	Transpose-sequential_1/conv2dmpo_1/Tensordot_1:output:0>sequential_1/conv2dmpo_1/Tensordot_2/transpose_1/perm:output:0*
T0*.
_output_shapes
:Е
4sequential_1/conv2dmpo_1/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   А   Ў
.sequential_1/conv2dmpo_1/Tensordot_2/Reshape_1Reshape4sequential_1/conv2dmpo_1/Tensordot_2/transpose_1:y:0=sequential_1/conv2dmpo_1/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А–
+sequential_1/conv2dmpo_1/Tensordot_2/MatMulMatMul5sequential_1/conv2dmpo_1/Tensordot_2/Reshape:output:07sequential_1/conv2dmpo_1/Tensordot_2/Reshape_1:output:0*
T0* 
_output_shapes
:
ААЫ
*sequential_1/conv2dmpo_1/Tensordot_2/shapeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                              д
$sequential_1/conv2dmpo_1/Tensordot_2Reshape5sequential_1/conv2dmpo_1/Tensordot_2/MatMul:product:03sequential_1/conv2dmpo_1/Tensordot_2/shape:output:0*
T0*>
_output_shapes,
*:(Ш
'sequential_1/conv2dmpo_1/transpose/permConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   ў
"sequential_1/conv2dmpo_1/transpose	Transpose-sequential_1/conv2dmpo_1/Tensordot_2:output:00sequential_1/conv2dmpo_1/transpose/perm:output:0*
T0*>
_output_shapes,
*:(Ъ
)sequential_1/conv2dmpo_1/transpose_1/permConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                	               ÷
$sequential_1/conv2dmpo_1/transpose_1	Transpose&sequential_1/conv2dmpo_1/transpose:y:02sequential_1/conv2dmpo_1/transpose_1/perm:output:0*
T0*>
_output_shapes,
*:(П
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
valueB: 
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
valueB: †
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
valueB:‘
(sequential_1/conv2dmpo_1/strided_slice_1StridedSlice'sequential_1/conv2dmpo_1/Shape:output:07sequential_1/conv2dmpo_1/strided_slice_1/stack:output:09sequential_1/conv2dmpo_1/strided_slice_1/stack_1:output:09sequential_1/conv2dmpo_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskЖ
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
€€€€€€€€€о
sequential_1/conv2dmpo_1/concatConcatV21sequential_1/conv2dmpo_1/strided_slice_1:output:01sequential_1/conv2dmpo_1/concat/values_1:output:0-sequential_1/conv2dmpo_1/concat/axis:output:0*
N*
T0*
_output_shapes
:Љ
 sequential_1/conv2dmpo_1/ReshapeReshape(sequential_1/conv2dmpo_1/transpose_1:y:0(sequential_1/conv2dmpo_1/concat:output:0*
T0*2
_output_shapes 
:О
)sequential_1/conv2dmpo_1/transpose_2/permConst*
_output_shapes
:*
dtype0*1
value(B&"                      Ќ
$sequential_1/conv2dmpo_1/transpose_2	Transpose)sequential_1/conv2dmpo_1/Reshape:output:02sequential_1/conv2dmpo_1/transpose_2/perm:output:0*
T0*2
_output_shapes 
:Е
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
valueB:‘
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
valueB: ¶
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
valueB:÷
(sequential_1/conv2dmpo_1/strided_slice_3StridedSlice)sequential_1/conv2dmpo_1/Shape_1:output:07sequential_1/conv2dmpo_1/strided_slice_3/stack:output:09sequential_1/conv2dmpo_1/strided_slice_3/stack_1:output:09sequential_1/conv2dmpo_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskК
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
€€€€€€€€€ф
!sequential_1/conv2dmpo_1/concat_1ConcatV21sequential_1/conv2dmpo_1/strided_slice_3:output:03sequential_1/conv2dmpo_1/concat_1/values_1:output:0/sequential_1/conv2dmpo_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:µ
"sequential_1/conv2dmpo_1/Reshape_1Reshape(sequential_1/conv2dmpo_1/transpose_2:y:0*sequential_1/conv2dmpo_1/concat_1:output:0*
T0*'
_output_shapes
:Аж
sequential_1/conv2dmpo_1/Conv2DConv2D+sequential_1/max_pooling2d/MaxPool:output:0+sequential_1/conv2dmpo_1/Reshape_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
©
1sequential_1/conv2dmpo_1/Reshape_2/ReadVariableOpReadVariableOp:sequential_1_conv2dmpo_1_reshape_2_readvariableop_resource*
_output_shapes	
:А*
dtype0y
(sequential_1/conv2dmpo_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ≈
"sequential_1/conv2dmpo_1/Reshape_2Reshape9sequential_1/conv2dmpo_1/Reshape_2/ReadVariableOp:value:01sequential_1/conv2dmpo_1/Reshape_2/shape:output:0*
T0*
_output_shapes
:	АЈ
sequential_1/conv2dmpo_1/addAddV2(sequential_1/conv2dmpo_1/Conv2D:output:0+sequential_1/conv2dmpo_1/Reshape_2:output:0*
T0*0
_output_shapes
:€€€€€€€€€АВ
sequential_1/conv2dmpo_1/ReluRelu sequential_1/conv2dmpo_1/add:z:0*
T0*0
_output_shapes
:€€€€€€€€€А•
3sequential_1/conv2dmpo_1/ActivityRegularizer/SquareSquare+sequential_1/conv2dmpo_1/Relu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€АЛ
2sequential_1/conv2dmpo_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ќ
0sequential_1/conv2dmpo_1/ActivityRegularizer/SumSum7sequential_1/conv2dmpo_1/ActivityRegularizer/Square:y:0;sequential_1/conv2dmpo_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: w
2sequential_1/conv2dmpo_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ђ≈'7–
0sequential_1/conv2dmpo_1/ActivityRegularizer/mulMul;sequential_1/conv2dmpo_1/ActivityRegularizer/mul/x:output:09sequential_1/conv2dmpo_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: Н
2sequential_1/conv2dmpo_1/ActivityRegularizer/ShapeShape+sequential_1/conv2dmpo_1/Relu:activations:0*
T0*
_output_shapes
:К
@sequential_1/conv2dmpo_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: М
Bsequential_1/conv2dmpo_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:М
Bsequential_1/conv2dmpo_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:≤
:sequential_1/conv2dmpo_1/ActivityRegularizer/strided_sliceStridedSlice;sequential_1/conv2dmpo_1/ActivityRegularizer/Shape:output:0Isequential_1/conv2dmpo_1/ActivityRegularizer/strided_slice/stack:output:0Ksequential_1/conv2dmpo_1/ActivityRegularizer/strided_slice/stack_1:output:0Ksequential_1/conv2dmpo_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЃ
1sequential_1/conv2dmpo_1/ActivityRegularizer/CastCastCsequential_1/conv2dmpo_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ќ
4sequential_1/conv2dmpo_1/ActivityRegularizer/truedivRealDiv4sequential_1/conv2dmpo_1/ActivityRegularizer/mul:z:05sequential_1/conv2dmpo_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
:  
$sequential_1/max_pooling2d_1/MaxPoolMaxPool+sequential_1/conv2dmpo_1/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
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
valueB:ƒ
$sequential_1/fruit_tn1/strided_sliceStridedSlice%sequential_1/fruit_tn1/Shape:output:03sequential_1/fruit_tn1/strided_slice/stack:output:05sequential_1/fruit_tn1/strided_slice/stack_1:output:05sequential_1/fruit_tn1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЗ
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
value	B :Ѕ
sequential_1/fruit_tn1/rangeRange+sequential_1/fruit_tn1/range/start:output:0$sequential_1/fruit_tn1/Rank:output:0+sequential_1/fruit_tn1/range/delta:output:0*
_output_shapes
:Е
 sequential_1/fruit_tn1/Max/inputPack-sequential_1/fruit_tn1/strided_slice:output:0*
N*
T0*
_output_shapes
:Ф
sequential_1/fruit_tn1/MaxMax)sequential_1/fruit_tn1/Max/input:output:0%sequential_1/fruit_tn1/range:output:0*
T0*
_output_shapes
: 
=sequential_1/fruit_tn1/loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : Ћ
7sequential_1/fruit_tn1/loop_body/PlaceholderWithDefaultPlaceholderWithDefaultFsequential_1/fruit_tn1/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: Г
&sequential_1/fruit_tn1/loop_body/ShapeShape-sequential_1/max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:~
4sequential_1/fruit_tn1/loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: А
6sequential_1/fruit_tn1/loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6sequential_1/fruit_tn1/loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
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
value	B :¬
(sequential_1/fruit_tn1/loop_body/GreaterGreater7sequential_1/fruit_tn1/loop_body/strided_slice:output:03sequential_1/fruit_tn1/loop_body/Greater/y:output:0*
T0*
_output_shapes
: m
+sequential_1/fruit_tn1/loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : ь
)sequential_1/fruit_tn1/loop_body/SelectV2SelectV2,sequential_1/fruit_tn1/loop_body/Greater:z:0@sequential_1/fruit_tn1/loop_body/PlaceholderWithDefault:output:04sequential_1/fruit_tn1/loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: p
.sequential_1/fruit_tn1/loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ґ
)sequential_1/fruit_tn1/loop_body/GatherV2GatherV2-sequential_1/max_pooling2d_1/MaxPool:output:02sequential_1/fruit_tn1/loop_body/SelectV2:output:07sequential_1/fruit_tn1/loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:Ађ
/sequential_1/fruit_tn1/loop_body/ReadVariableOpReadVariableOp8sequential_1_fruit_tn1_loop_body_readvariableop_resource*"
_output_shapes
:*
dtype0µ
1sequential_1/fruit_tn1/loop_body/ReadVariableOp_1ReadVariableOp:sequential_1_fruit_tn1_loop_body_readvariableop_1_resource*'
_output_shapes
:А*
dtype0∞
1sequential_1/fruit_tn1/loop_body/ReadVariableOp_2ReadVariableOp:sequential_1_fruit_tn1_loop_body_readvariableop_2_resource*"
_output_shapes
:*
dtype0О
9sequential_1/fruit_tn1/loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          з
4sequential_1/fruit_tn1/loop_body/Tensordot/transpose	Transpose2sequential_1/fruit_tn1/loop_body/GatherV2:output:0Bsequential_1/fruit_tn1/loop_body/Tensordot/transpose/perm:output:0*
T0*#
_output_shapes
:АЙ
8sequential_1/fruit_tn1/loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      д
2sequential_1/fruit_tn1/loop_body/Tensordot/ReshapeReshape8sequential_1/fruit_tn1/loop_body/Tensordot/transpose:y:0Asequential_1/fruit_tn1/loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	АР
;sequential_1/fruit_tn1/loop_body/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          п
6sequential_1/fruit_tn1/loop_body/Tensordot/transpose_1	Transpose7sequential_1/fruit_tn1/loop_body/ReadVariableOp:value:0Dsequential_1/fruit_tn1/loop_body/Tensordot/transpose_1/perm:output:0*
T0*"
_output_shapes
:Л
:sequential_1/fruit_tn1/loop_body/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      й
4sequential_1/fruit_tn1/loop_body/Tensordot/Reshape_1Reshape:sequential_1/fruit_tn1/loop_body/Tensordot/transpose_1:y:0Csequential_1/fruit_tn1/loop_body/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:б
1sequential_1/fruit_tn1/loop_body/Tensordot/MatMulMatMul;sequential_1/fruit_tn1/loop_body/Tensordot/Reshape:output:0=sequential_1/fruit_tn1/loop_body/Tensordot/Reshape_1:output:0*
T0*
_output_shapes
:	АЙ
0sequential_1/fruit_tn1/loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            я
*sequential_1/fruit_tn1/loop_body/TensordotReshape;sequential_1/fruit_tn1/loop_body/Tensordot/MatMul:product:09sequential_1/fruit_tn1/loop_body/Tensordot/shape:output:0*
T0*'
_output_shapes
:АР
;sequential_1/fruit_tn1/loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          с
6sequential_1/fruit_tn1/loop_body/Tensordot_1/transpose	Transpose9sequential_1/fruit_tn1/loop_body/ReadVariableOp_2:value:0Dsequential_1/fruit_tn1/loop_body/Tensordot_1/transpose/perm:output:0*
T0*"
_output_shapes
:Л
:sequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      й
4sequential_1/fruit_tn1/loop_body/Tensordot_1/ReshapeReshape:sequential_1/fruit_tn1/loop_body/Tensordot_1/transpose:y:0Csequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:Н
<sequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      з
6sequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1Reshape3sequential_1/fruit_tn1/loop_body/Tensordot:output:0Esequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А(з
3sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMulMatMul=sequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape:output:0?sequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	А(П
2sequential_1/fruit_tn1/loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"               й
,sequential_1/fruit_tn1/loop_body/Tensordot_1Reshape=sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul:product:0;sequential_1/fruit_tn1/loop_body/Tensordot_1/shape:output:0*
T0*+
_output_shapes
:АЛ
:sequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      й
4sequential_1/fruit_tn1/loop_body/Tensordot_2/ReshapeReshape9sequential_1/fruit_tn1/loop_body/ReadVariableOp_1:value:0Csequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes
:	А2Ш
;sequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ц
6sequential_1/fruit_tn1/loop_body/Tensordot_2/transpose	Transpose5sequential_1/fruit_tn1/loop_body/Tensordot_1:output:0Dsequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/perm:output:0*
T0*+
_output_shapes
:АН
<sequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      о
6sequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1Reshape:sequential_1/fruit_tn1/loop_body/Tensordot_2/transpose:y:0Esequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А2ж
3sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMulMatMul=sequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape:output:0?sequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1:output:0*
T0*
_output_shapes

:З
2sequential_1/fruit_tn1/loop_body/Tensordot_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         а
,sequential_1/fruit_tn1/loop_body/Tensordot_2Reshape=sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul:product:0;sequential_1/fruit_tn1/loop_body/Tensordot_2/shape:output:0*
T0*"
_output_shapes
:Д
/sequential_1/fruit_tn1/loop_body/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ’
*sequential_1/fruit_tn1/loop_body/transpose	Transpose5sequential_1/fruit_tn1/loop_body/Tensordot_2:output:08sequential_1/fruit_tn1/loop_body/transpose/perm:output:0*
T0*"
_output_shapes
:і
3sequential_1/fruit_tn1/loop_body/add/ReadVariableOpReadVariableOp<sequential_1_fruit_tn1_loop_body_add_readvariableop_resource*"
_output_shapes
:*
dtype0«
$sequential_1/fruit_tn1/loop_body/addAddV2.sequential_1/fruit_tn1/loop_body/transpose:y:0;sequential_1/fruit_tn1/loop_body/add/ReadVariableOp:value:0*
T0*"
_output_shapes
:s
)sequential_1/fruit_tn1/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:ђ
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
value	B :Ў
!sequential_1/fruit_tn1/pfor/rangeRange0sequential_1/fruit_tn1/pfor/range/start:output:0#sequential_1/fruit_tn1/Max:output:00sequential_1/fruit_tn1/pfor/range/delta:output:0*#
_output_shapes
:€€€€€€€€€u
3sequential_1/fruit_tn1/loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : v
4sequential_1/fruit_tn1/loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :ў
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
value	B :я
4sequential_1/fruit_tn1/loop_body/SelectV2/pfor/add_1AddV2>sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Rank_2:output:0?sequential_1/fruit_tn1/loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: Џ
6sequential_1/fruit_tn1/loop_body/SelectV2/pfor/MaximumMaximum>sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Rank_1:output:06sequential_1/fruit_tn1/loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: Џ
8sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Maximum_1Maximum8sequential_1/fruit_tn1/loop_body/SelectV2/pfor/add_1:z:0:sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: О
4sequential_1/fruit_tn1/loop_body/SelectV2/pfor/ShapeShape*sequential_1/fruit_tn1/pfor/range:output:0*
T0*
_output_shapes
:Ў
2sequential_1/fruit_tn1/loop_body/SelectV2/pfor/subSub<sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Maximum_1:z:0>sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: Ж
<sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:е
6sequential_1/fruit_tn1/loop_body/SelectV2/pfor/ReshapeReshape6sequential_1/fruit_tn1/loop_body/SelectV2/pfor/sub:z:0Esequential_1/fruit_tn1/loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:Г
9sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:г
3sequential_1/fruit_tn1/loop_body/SelectV2/pfor/TileTileBsequential_1/fruit_tn1/loop_body/SelectV2/pfor/Tile/input:output:0?sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: М
Bsequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: О
Dsequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:О
Dsequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
<sequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_sliceStridedSlice=sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Shape:output:0Ksequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack:output:0Msequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_1:output:0Msequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskО
Dsequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Р
Fsequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Р
Fsequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Њ
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
value	B : В
5sequential_1/fruit_tn1/loop_body/SelectV2/pfor/concatConcatV2Esequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice:output:0<sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Tile:output:0Gsequential_1/fruit_tn1/loop_body/SelectV2/pfor/strided_slice_1:output:0Csequential_1/fruit_tn1/loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ё
8sequential_1/fruit_tn1/loop_body/SelectV2/pfor/Reshape_1Reshape*sequential_1/fruit_tn1/pfor/range:output:0>sequential_1/fruit_tn1/loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:€€€€€€€€€Ш
7sequential_1/fruit_tn1/loop_body/SelectV2/pfor/SelectV2SelectV2,sequential_1/fruit_tn1/loop_body/Greater:z:0Asequential_1/fruit_tn1/loop_body/SelectV2/pfor/Reshape_1:output:04sequential_1/fruit_tn1/loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:€€€€€€€€€~
<sequential_1/fruit_tn1/loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ў
7sequential_1/fruit_tn1/loop_body/GatherV2/pfor/GatherV2GatherV2-sequential_1/max_pooling2d_1/MaxPool:output:0@sequential_1/fruit_tn1/loop_body/SelectV2/pfor/SelectV2:output:0Esequential_1/fruit_tn1/loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*0
_output_shapes
:€€€€€€€€€АБ
?sequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :щ
=sequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/addAddV2Bsequential_1/fruit_tn1/loop_body/Tensordot/transpose/perm:output:0Hsequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:У
Isequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: З
Esequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : б
@sequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/concatConcatV2Rsequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/concat/values_0:output:0Asequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/add:z:0Nsequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ш
Csequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/Transpose	Transpose@sequential_1/fruit_tn1/loop_body/GatherV2/pfor/GatherV2:output:0Isequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЕ
Csequential_1/fruit_tn1/loop_body/Tensordot/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ј
>sequential_1/fruit_tn1/loop_body/Tensordot/Reshape/pfor/concatConcatV2,sequential_1/fruit_tn1/pfor/Reshape:output:0Asequential_1/fruit_tn1/loop_body/Tensordot/Reshape/shape:output:0Lsequential_1/fruit_tn1/loop_body/Tensordot/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:У
?sequential_1/fruit_tn1/loop_body/Tensordot/Reshape/pfor/ReshapeReshapeGsequential_1/fruit_tn1/loop_body/Tensordot/transpose/pfor/Transpose:y:0Gsequential_1/fruit_tn1/loop_body/Tensordot/Reshape/pfor/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Аі
<sequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/ShapeShapeHsequential_1/fruit_tn1/loop_body/Tensordot/Reshape/pfor/Reshape:output:0*
T0*
_output_shapes
:И
Fsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Я
<sequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/splitSplitOsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/split/split_dim:output:0Esequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Shape:output:0*
T0*&
_output_shapes
:::*
	num_splitЗ
Dsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB Й
Fsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB В
>sequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/ReshapeReshapeEsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/split:output:0Osequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1:output:0*
T0*
_output_shapes
: Й
Fsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB Л
Hsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB Ж
@sequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1ReshapeEsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/split:output:1Qsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: Й
Fsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB Л
Hsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB Ж
@sequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2ReshapeEsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/split:output:2Qsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ц
:sequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/mulMulGsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape:output:0Isequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: З
Fsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack>sequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/mul:z:0Isequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:°
@sequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3ReshapeHsequential_1/fruit_tn1/loop_body/Tensordot/Reshape/pfor/Reshape:output:0Osequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€Г
=sequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/MatMulMatMulIsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0=sequential_1/fruit_tn1/loop_body/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€У
Hsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€г
Fsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePackGsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape:output:0Isequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0Qsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:Ь
@sequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4ReshapeGsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/MatMul:product:0Osequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€А}
;sequential_1/fruit_tn1/loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
6sequential_1/fruit_tn1/loop_body/Tensordot/pfor/concatConcatV2,sequential_1/fruit_tn1/pfor/Reshape:output:09sequential_1/fruit_tn1/loop_body/Tensordot/shape:output:0Dsequential_1/fruit_tn1/loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Н
7sequential_1/fruit_tn1/loop_body/Tensordot/pfor/ReshapeReshapeIsequential_1/fruit_tn1/loop_body/Tensordot/MatMul/pfor/Reshape_4:output:0?sequential_1/fruit_tn1/loop_body/Tensordot/pfor/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АЙ
Gsequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : √
Bsequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2,sequential_1/fruit_tn1/pfor/Reshape:output:0Esequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1/shape:output:0Psequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ф
Csequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape@sequential_1/fruit_tn1/loop_body/Tensordot/pfor/Reshape:output:0Ksequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А(Ї
>sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/ShapeShapeLsequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:Ц
Lsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ш
Nsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ш
Nsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:о
Fsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSliceGsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Usequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskШ
Nsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
Hsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSliceGsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
@sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/MinimumMinimumOsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0Qsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: В
@sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :щ
>sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/EqualEqualDsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Minimum:z:0Isequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: Ц
Asequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
Asequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"           
?sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/SelectSelectBsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Equal:z:0Jsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/t:output:0Jsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:Ш
Nsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
Hsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSliceGsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskШ
Nsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: Ъ
Psequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
Hsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSliceGsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskШ
Nsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
Hsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSliceGsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskн
>sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/stackPackQsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0Qsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0Qsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:ѓ
Bsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose	TransposeLsequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0Hsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€У
@sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/ReshapeReshapeFsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose:y:0Gsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:€€€€€€€€€А(є
@sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape_1ShapeIsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:К
Hsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : І
>sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/splitSplitQsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split/split_dim:output:0Isequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_splitЛ
Hsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB Н
Jsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB М
Bsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1ReshapeGsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split:output:0Ssequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: Л
Hsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB Н
Jsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB М
Bsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2ReshapeGsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split:output:1Ssequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: Л
Hsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB Н
Jsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB М
Bsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3ReshapeGsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/split:output:2Ssequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ю
<sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/mulMulKsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0Ksequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: Н
Hsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePackKsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:0@sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:¶
Bsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4ReshapeIsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0Qsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€З
?sequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul=sequential_1/fruit_tn1/loop_body/Tensordot_1/Reshape:output:0Ksequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€Х
Jsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€н
Hsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePackSsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:0Ksequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0Ksequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:≥
Bsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5ReshapeIsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0Qsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€Ю
Isequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ©
Dsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1	TransposeKsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0Rsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€А(
=sequential_1/fruit_tn1/loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : •
8sequential_1/fruit_tn1/loop_body/Tensordot_1/pfor/concatConcatV2,sequential_1/fruit_tn1/pfor/Reshape:output:0;sequential_1/fruit_tn1/loop_body/Tensordot_1/shape:output:0Fsequential_1/fruit_tn1/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ф
9sequential_1/fruit_tn1/loop_body/Tensordot_1/pfor/ReshapeReshapeHsequential_1/fruit_tn1/loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:0Asequential_1/fruit_tn1/loop_body/Tensordot_1/pfor/concat:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€АГ
Asequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :€
?sequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/addAddV2Dsequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/perm:output:0Jsequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:Х
Ksequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: Й
Gsequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : й
Bsequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concatConcatV2Tsequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/values_0:output:0Csequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/add:z:0Psequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:¶
Esequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/Transpose	TransposeBsequential_1/fruit_tn1/loop_body/Tensordot_1/pfor/Reshape:output:0Ksequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/concat:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€АЙ
Gsequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : √
Bsequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concatConcatV2,sequential_1/fruit_tn1/pfor/Reshape:output:0Esequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1/shape:output:0Psequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Э
Csequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/ReshapeReshapeIsequential_1/fruit_tn1/loop_body/Tensordot_2/transpose/pfor/Transpose:y:0Ksequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2Ї
>sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/ShapeShapeLsequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:Ц
Lsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ш
Nsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ш
Nsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:о
Fsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_sliceStridedSliceGsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Usequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskШ
Nsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
Hsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1StridedSliceGsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
@sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/MinimumMinimumOsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice:output:0Qsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: В
@sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :щ
>sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/EqualEqualDsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Minimum:z:0Isequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: Ц
Asequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
Asequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"           
?sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/SelectSelectBsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Equal:z:0Jsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/t:output:0Jsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:Ш
Nsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
Hsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2StridedSliceGsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskШ
Nsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: Ъ
Psequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
Hsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3StridedSliceGsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskШ
Nsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
Hsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4StridedSliceGsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1:output:0Ysequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskн
>sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/stackPackQsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2:output:0Qsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3:output:0Qsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:ѓ
Bsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose	TransposeLsequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0Hsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€У
@sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/ReshapeReshapeFsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose:y:0Gsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:А2€€€€€€€€€є
@sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape_1ShapeIsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:К
Hsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : І
>sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/splitSplitQsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split/split_dim:output:0Isequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_splitЛ
Hsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB Н
Jsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB М
Bsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1ReshapeGsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split:output:0Ssequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: Л
Hsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB Н
Jsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB М
Bsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2ReshapeGsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split:output:1Ssequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: Л
Hsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB Н
Jsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB М
Bsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3ReshapeGsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/split:output:2Ssequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ю
<sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/mulMulKsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0Ksequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: Н
Hsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shapePackKsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_1:output:0@sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:¶
Bsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4ReshapeIsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0Qsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€З
?sequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/MatMulMatMul=sequential_1/fruit_tn1/loop_body/Tensordot_2/Reshape:output:0Ksequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€Х
Jsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€н
Hsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shapePackSsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0:output:0Ksequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0Ksequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:≥
Bsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5ReshapeIsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/MatMul:product:0Qsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€Ю
Isequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
Dsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1	TransposeKsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/Reshape_5:output:0Rsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€
=sequential_1/fruit_tn1/loop_body/Tensordot_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : •
8sequential_1/fruit_tn1/loop_body/Tensordot_2/pfor/concatConcatV2,sequential_1/fruit_tn1/pfor/Reshape:output:0;sequential_1/fruit_tn1/loop_body/Tensordot_2/shape:output:0Fsequential_1/fruit_tn1/loop_body/Tensordot_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Л
9sequential_1/fruit_tn1/loop_body/Tensordot_2/pfor/ReshapeReshapeHsequential_1/fruit_tn1/loop_body/Tensordot_2/MatMul/pfor/transpose_1:y:0Asequential_1/fruit_tn1/loop_body/Tensordot_2/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€w
5sequential_1/fruit_tn1/loop_body/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :џ
3sequential_1/fruit_tn1/loop_body/transpose/pfor/addAddV28sequential_1/fruit_tn1/loop_body/transpose/perm:output:0>sequential_1/fruit_tn1/loop_body/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:Й
?sequential_1/fruit_tn1/loop_body/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: }
;sequential_1/fruit_tn1/loop_body/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : є
6sequential_1/fruit_tn1/loop_body/transpose/pfor/concatConcatV2Hsequential_1/fruit_tn1/loop_body/transpose/pfor/concat/values_0:output:07sequential_1/fruit_tn1/loop_body/transpose/pfor/add:z:0Dsequential_1/fruit_tn1/loop_body/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Е
9sequential_1/fruit_tn1/loop_body/transpose/pfor/Transpose	TransposeBsequential_1/fruit_tn1/loop_body/Tensordot_2/pfor/Reshape:output:0?sequential_1/fruit_tn1/loop_body/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€p
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
value	B :ћ
-sequential_1/fruit_tn1/loop_body/add/pfor/addAddV29sequential_1/fruit_tn1/loop_body/add/pfor/Rank_1:output:08sequential_1/fruit_tn1/loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: …
1sequential_1/fruit_tn1/loop_body/add/pfor/MaximumMaximum1sequential_1/fruit_tn1/loop_body/add/pfor/add:z:07sequential_1/fruit_tn1/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: Ь
/sequential_1/fruit_tn1/loop_body/add/pfor/ShapeShape=sequential_1/fruit_tn1/loop_body/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:≈
-sequential_1/fruit_tn1/loop_body/add/pfor/subSub5sequential_1/fruit_tn1/loop_body/add/pfor/Maximum:z:07sequential_1/fruit_tn1/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: Б
7sequential_1/fruit_tn1/loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:÷
1sequential_1/fruit_tn1/loop_body/add/pfor/ReshapeReshape1sequential_1/fruit_tn1/loop_body/add/pfor/sub:z:0@sequential_1/fruit_tn1/loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:~
4sequential_1/fruit_tn1/loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:‘
.sequential_1/fruit_tn1/loop_body/add/pfor/TileTile=sequential_1/fruit_tn1/loop_body/add/pfor/Tile/input:output:0:sequential_1/fruit_tn1/loop_body/add/pfor/Reshape:output:0*
T0*
_output_shapes
: З
=sequential_1/fruit_tn1/loop_body/add/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Й
?sequential_1/fruit_tn1/loop_body/add/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Й
?sequential_1/fruit_tn1/loop_body/add/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
7sequential_1/fruit_tn1/loop_body/add/pfor/strided_sliceStridedSlice8sequential_1/fruit_tn1/loop_body/add/pfor/Shape:output:0Fsequential_1/fruit_tn1/loop_body/add/pfor/strided_slice/stack:output:0Hsequential_1/fruit_tn1/loop_body/add/pfor/strided_slice/stack_1:output:0Hsequential_1/fruit_tn1/loop_body/add/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskЙ
?sequential_1/fruit_tn1/loop_body/add/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Л
Asequential_1/fruit_tn1/loop_body/add/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Л
Asequential_1/fruit_tn1/loop_body/add/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:І
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
value	B : й
0sequential_1/fruit_tn1/loop_body/add/pfor/concatConcatV2@sequential_1/fruit_tn1/loop_body/add/pfor/strided_slice:output:07sequential_1/fruit_tn1/loop_body/add/pfor/Tile:output:0Bsequential_1/fruit_tn1/loop_body/add/pfor/strided_slice_1:output:0>sequential_1/fruit_tn1/loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:т
3sequential_1/fruit_tn1/loop_body/add/pfor/Reshape_1Reshape=sequential_1/fruit_tn1/loop_body/transpose/pfor/Transpose:y:09sequential_1/fruit_tn1/loop_body/add/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€н
/sequential_1/fruit_tn1/loop_body/add/pfor/AddV2AddV2<sequential_1/fruit_tn1/loop_body/add/pfor/Reshape_1:output:0;sequential_1/fruit_tn1/loop_body/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€u
$sequential_1/fruit_tn1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   њ
sequential_1/fruit_tn1/ReshapeReshape3sequential_1/fruit_tn1/loop_body/add/pfor/AddV2:z:0-sequential_1/fruit_tn1/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@~
sequential_1/fruit_tn1/ReluRelu'sequential_1/fruit_tn1/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ш
1sequential_1/fruit_tn1/ActivityRegularizer/SquareSquare)sequential_1/fruit_tn1/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@Б
0sequential_1/fruit_tn1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       »
.sequential_1/fruit_tn1/ActivityRegularizer/SumSum5sequential_1/fruit_tn1/ActivityRegularizer/Square:y:09sequential_1/fruit_tn1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: u
0sequential_1/fruit_tn1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ђ≈'7 
.sequential_1/fruit_tn1/ActivityRegularizer/mulMul9sequential_1/fruit_tn1/ActivityRegularizer/mul/x:output:07sequential_1/fruit_tn1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: Й
0sequential_1/fruit_tn1/ActivityRegularizer/ShapeShape)sequential_1/fruit_tn1/Relu:activations:0*
T0*
_output_shapes
:И
>sequential_1/fruit_tn1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: К
@sequential_1/fruit_tn1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:К
@sequential_1/fruit_tn1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
8sequential_1/fruit_tn1/ActivityRegularizer/strided_sliceStridedSlice9sequential_1/fruit_tn1/ActivityRegularizer/Shape:output:0Gsequential_1/fruit_tn1/ActivityRegularizer/strided_slice/stack:output:0Isequential_1/fruit_tn1/ActivityRegularizer/strided_slice/stack_1:output:0Isequential_1/fruit_tn1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask™
/sequential_1/fruit_tn1/ActivityRegularizer/CastCastAsequential_1/fruit_tn1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: «
2sequential_1/fruit_tn1/ActivityRegularizer/truedivRealDiv2sequential_1/fruit_tn1/ActivityRegularizer/mul:z:03sequential_1/fruit_tn1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: З
sequential_1/dropout1/IdentityIdentity)sequential_1/fruit_tn1/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@s
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
valueB:ƒ
$sequential_1/fruit_tn2/strided_sliceStridedSlice%sequential_1/fruit_tn2/Shape:output:03sequential_1/fruit_tn2/strided_slice/stack:output:05sequential_1/fruit_tn2/strided_slice/stack_1:output:05sequential_1/fruit_tn2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЗ
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
value	B :Ѕ
sequential_1/fruit_tn2/rangeRange+sequential_1/fruit_tn2/range/start:output:0$sequential_1/fruit_tn2/Rank:output:0+sequential_1/fruit_tn2/range/delta:output:0*
_output_shapes
:Е
 sequential_1/fruit_tn2/Max/inputPack-sequential_1/fruit_tn2/strided_slice:output:0*
N*
T0*
_output_shapes
:Ф
sequential_1/fruit_tn2/MaxMax)sequential_1/fruit_tn2/Max/input:output:0%sequential_1/fruit_tn2/range:output:0*
T0*
_output_shapes
: 
=sequential_1/fruit_tn2/loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : Ћ
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
valueB: А
6sequential_1/fruit_tn2/loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6sequential_1/fruit_tn2/loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
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
value	B :¬
(sequential_1/fruit_tn2/loop_body/GreaterGreater7sequential_1/fruit_tn2/loop_body/strided_slice:output:03sequential_1/fruit_tn2/loop_body/Greater/y:output:0*
T0*
_output_shapes
: m
+sequential_1/fruit_tn2/loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : ь
)sequential_1/fruit_tn2/loop_body/SelectV2SelectV2,sequential_1/fruit_tn2/loop_body/Greater:z:0@sequential_1/fruit_tn2/loop_body/PlaceholderWithDefault:output:04sequential_1/fruit_tn2/loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: p
.sequential_1/fruit_tn2/loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : У
)sequential_1/fruit_tn2/loop_body/GatherV2GatherV2'sequential_1/dropout1/Identity:output:02sequential_1/fruit_tn2/loop_body/SelectV2:output:07sequential_1/fruit_tn2/loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:@Г
.sequential_1/fruit_tn2/loop_body/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         Ќ
(sequential_1/fruit_tn2/loop_body/ReshapeReshape2sequential_1/fruit_tn2/loop_body/GatherV2:output:07sequential_1/fruit_tn2/loop_body/Reshape/shape:output:0*
T0*"
_output_shapes
:®
/sequential_1/fruit_tn2/loop_body/ReadVariableOpReadVariableOp8sequential_1_fruit_tn2_loop_body_readvariableop_resource*
_output_shapes

:*
dtype0µ
1sequential_1/fruit_tn2/loop_body/ReadVariableOp_1ReadVariableOp:sequential_1_fruit_tn2_loop_body_readvariableop_1_resource*'
_output_shapes
:Г*
dtype0ђ
1sequential_1/fruit_tn2/loop_body/ReadVariableOp_2ReadVariableOp:sequential_1_fruit_tn2_loop_body_readvariableop_2_resource*
_output_shapes

:*
dtype0О
9sequential_1/fruit_tn2/loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          е
4sequential_1/fruit_tn2/loop_body/Tensordot/transpose	Transpose1sequential_1/fruit_tn2/loop_body/Reshape:output:0Bsequential_1/fruit_tn2/loop_body/Tensordot/transpose/perm:output:0*
T0*"
_output_shapes
:Й
8sequential_1/fruit_tn2/loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      г
2sequential_1/fruit_tn2/loop_body/Tensordot/ReshapeReshape8sequential_1/fruit_tn2/loop_body/Tensordot/transpose:y:0Asequential_1/fruit_tn2/loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:Џ
1sequential_1/fruit_tn2/loop_body/Tensordot/MatMulMatMul;sequential_1/fruit_tn2/loop_body/Tensordot/Reshape:output:07sequential_1/fruit_tn2/loop_body/ReadVariableOp:value:0*
T0*
_output_shapes

:Е
0sequential_1/fruit_tn2/loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         Џ
*sequential_1/fruit_tn2/loop_body/TensordotReshape;sequential_1/fruit_tn2/loop_body/Tensordot/MatMul:product:09sequential_1/fruit_tn2/loop_body/Tensordot/shape:output:0*
T0*"
_output_shapes
:Ф
;sequential_1/fruit_tn2/loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ц
6sequential_1/fruit_tn2/loop_body/Tensordot_1/transpose	Transpose9sequential_1/fruit_tn2/loop_body/ReadVariableOp_1:value:0Dsequential_1/fruit_tn2/loop_body/Tensordot_1/transpose/perm:output:0*
T0*'
_output_shapes
:ГЛ
:sequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"     к
4sequential_1/fruit_tn2/loop_body/Tensordot_1/ReshapeReshape:sequential_1/fruit_tn2/loop_body/Tensordot_1/transpose:y:0Csequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes
:	МТ
=sequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          п
8sequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1	Transpose3sequential_1/fruit_tn2/loop_body/Tensordot:output:0Fsequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/perm:output:0*
T0*"
_output_shapes
:Н
<sequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      п
6sequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1Reshape<sequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1:y:0Esequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:з
3sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMulMatMul=sequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape:output:0?sequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	МЗ
2sequential_1/fruit_tn2/loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"Г         б
,sequential_1/fruit_tn2/loop_body/Tensordot_1Reshape=sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul:product:0;sequential_1/fruit_tn2/loop_body/Tensordot_1/shape:output:0*
T0*#
_output_shapes
:ГЛ
:sequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      и
4sequential_1/fruit_tn2/loop_body/Tensordot_2/ReshapeReshape9sequential_1/fruit_tn2/loop_body/ReadVariableOp_2:value:0Csequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes

:Р
;sequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          о
6sequential_1/fruit_tn2/loop_body/Tensordot_2/transpose	Transpose5sequential_1/fruit_tn2/loop_body/Tensordot_1:output:0Dsequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/perm:output:0*
T0*#
_output_shapes
:ГН
<sequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   Г   о
6sequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1Reshape:sequential_1/fruit_tn2/loop_body/Tensordot_2/transpose:y:0Esequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	Гз
3sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMulMatMul=sequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape:output:0?sequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1:output:0*
T0*
_output_shapes
:	Г}
2sequential_1/fruit_tn2/loop_body/Tensordot_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:Гў
,sequential_1/fruit_tn2/loop_body/Tensordot_2Reshape=sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul:product:0;sequential_1/fruit_tn2/loop_body/Tensordot_2/shape:output:0*
T0*
_output_shapes	
:Г≠
3sequential_1/fruit_tn2/loop_body/add/ReadVariableOpReadVariableOp<sequential_1_fruit_tn2_loop_body_add_readvariableop_resource*
_output_shapes	
:Г*
dtype0«
$sequential_1/fruit_tn2/loop_body/addAddV25sequential_1/fruit_tn2/loop_body/Tensordot_2:output:0;sequential_1/fruit_tn2/loop_body/add/ReadVariableOp:value:0*
T0*
_output_shapes	
:Гs
)sequential_1/fruit_tn2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:ђ
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
value	B :Ў
!sequential_1/fruit_tn2/pfor/rangeRange0sequential_1/fruit_tn2/pfor/range/start:output:0#sequential_1/fruit_tn2/Max:output:00sequential_1/fruit_tn2/pfor/range/delta:output:0*#
_output_shapes
:€€€€€€€€€u
3sequential_1/fruit_tn2/loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : v
4sequential_1/fruit_tn2/loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :ў
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
value	B :я
4sequential_1/fruit_tn2/loop_body/SelectV2/pfor/add_1AddV2>sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Rank_2:output:0?sequential_1/fruit_tn2/loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: Џ
6sequential_1/fruit_tn2/loop_body/SelectV2/pfor/MaximumMaximum>sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Rank_1:output:06sequential_1/fruit_tn2/loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: Џ
8sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Maximum_1Maximum8sequential_1/fruit_tn2/loop_body/SelectV2/pfor/add_1:z:0:sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: О
4sequential_1/fruit_tn2/loop_body/SelectV2/pfor/ShapeShape*sequential_1/fruit_tn2/pfor/range:output:0*
T0*
_output_shapes
:Ў
2sequential_1/fruit_tn2/loop_body/SelectV2/pfor/subSub<sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Maximum_1:z:0>sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: Ж
<sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:е
6sequential_1/fruit_tn2/loop_body/SelectV2/pfor/ReshapeReshape6sequential_1/fruit_tn2/loop_body/SelectV2/pfor/sub:z:0Esequential_1/fruit_tn2/loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:Г
9sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:г
3sequential_1/fruit_tn2/loop_body/SelectV2/pfor/TileTileBsequential_1/fruit_tn2/loop_body/SelectV2/pfor/Tile/input:output:0?sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: М
Bsequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: О
Dsequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:О
Dsequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
<sequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_sliceStridedSlice=sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Shape:output:0Ksequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack:output:0Msequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_1:output:0Msequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskО
Dsequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Р
Fsequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Р
Fsequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Њ
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
value	B : В
5sequential_1/fruit_tn2/loop_body/SelectV2/pfor/concatConcatV2Esequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice:output:0<sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Tile:output:0Gsequential_1/fruit_tn2/loop_body/SelectV2/pfor/strided_slice_1:output:0Csequential_1/fruit_tn2/loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ё
8sequential_1/fruit_tn2/loop_body/SelectV2/pfor/Reshape_1Reshape*sequential_1/fruit_tn2/pfor/range:output:0>sequential_1/fruit_tn2/loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:€€€€€€€€€Ш
7sequential_1/fruit_tn2/loop_body/SelectV2/pfor/SelectV2SelectV2,sequential_1/fruit_tn2/loop_body/Greater:z:0Asequential_1/fruit_tn2/loop_body/SelectV2/pfor/Reshape_1:output:04sequential_1/fruit_tn2/loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:€€€€€€€€€~
<sequential_1/fruit_tn2/loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :  
7sequential_1/fruit_tn2/loop_body/GatherV2/pfor/GatherV2GatherV2'sequential_1/dropout1/Identity:output:0@sequential_1/fruit_tn2/loop_body/SelectV2/pfor/SelectV2:output:0Esequential_1/fruit_tn2/loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:€€€€€€€€€@{
9sequential_1/fruit_tn2/loop_body/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Щ
4sequential_1/fruit_tn2/loop_body/Reshape/pfor/concatConcatV2,sequential_1/fruit_tn2/pfor/Reshape:output:07sequential_1/fruit_tn2/loop_body/Reshape/shape:output:0Bsequential_1/fruit_tn2/loop_body/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ы
5sequential_1/fruit_tn2/loop_body/Reshape/pfor/ReshapeReshape@sequential_1/fruit_tn2/loop_body/GatherV2/pfor/GatherV2:output:0=sequential_1/fruit_tn2/loop_body/Reshape/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€Б
?sequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :щ
=sequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/addAddV2Bsequential_1/fruit_tn2/loop_body/Tensordot/transpose/perm:output:0Hsequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:У
Isequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: З
Esequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : б
@sequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/concatConcatV2Rsequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/concat/values_0:output:0Asequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/add:z:0Nsequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Х
Csequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/Transpose	Transpose>sequential_1/fruit_tn2/loop_body/Reshape/pfor/Reshape:output:0Isequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€Е
Csequential_1/fruit_tn2/loop_body/Tensordot/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ј
>sequential_1/fruit_tn2/loop_body/Tensordot/Reshape/pfor/concatConcatV2,sequential_1/fruit_tn2/pfor/Reshape:output:0Asequential_1/fruit_tn2/loop_body/Tensordot/Reshape/shape:output:0Lsequential_1/fruit_tn2/loop_body/Tensordot/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Т
?sequential_1/fruit_tn2/loop_body/Tensordot/Reshape/pfor/ReshapeReshapeGsequential_1/fruit_tn2/loop_body/Tensordot/transpose/pfor/Transpose:y:0Gsequential_1/fruit_tn2/loop_body/Tensordot/Reshape/pfor/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€і
<sequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/ShapeShapeHsequential_1/fruit_tn2/loop_body/Tensordot/Reshape/pfor/Reshape:output:0*
T0*
_output_shapes
:И
Fsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Я
<sequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/splitSplitOsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/split/split_dim:output:0Esequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Shape:output:0*
T0*&
_output_shapes
:::*
	num_splitЗ
Dsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB Й
Fsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB В
>sequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/ReshapeReshapeEsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/split:output:0Osequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape/shape_1:output:0*
T0*
_output_shapes
: Й
Fsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB Л
Hsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB Ж
@sequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1ReshapeEsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/split:output:1Qsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: Й
Fsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB Л
Hsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB Ж
@sequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2ReshapeEsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/split:output:2Qsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ц
:sequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/mulMulGsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape:output:0Isequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: З
Fsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack>sequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/mul:z:0Isequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:°
@sequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3ReshapeHsequential_1/fruit_tn2/loop_body/Tensordot/Reshape/pfor/Reshape:output:0Osequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€э
=sequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/MatMulMatMulIsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_3:output:07sequential_1/fruit_tn2/loop_body/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€У
Hsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€г
Fsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePackGsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape:output:0Isequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0Qsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:Ы
@sequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4ReshapeGsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/MatMul:product:0Osequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€}
;sequential_1/fruit_tn2/loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
6sequential_1/fruit_tn2/loop_body/Tensordot/pfor/concatConcatV2,sequential_1/fruit_tn2/pfor/Reshape:output:09sequential_1/fruit_tn2/loop_body/Tensordot/shape:output:0Dsequential_1/fruit_tn2/loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:И
7sequential_1/fruit_tn2/loop_body/Tensordot/pfor/ReshapeReshapeIsequential_1/fruit_tn2/loop_body/Tensordot/MatMul/pfor/Reshape_4:output:0?sequential_1/fruit_tn2/loop_body/Tensordot/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€Е
Csequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Е
Asequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/addAddV2Fsequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/perm:output:0Lsequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/add/y:output:0*
T0*
_output_shapes
:Ч
Msequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: Л
Isequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : с
Dsequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concatConcatV2Vsequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/values_0:output:0Esequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/add:z:0Rsequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Я
Gsequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/Transpose	Transpose@sequential_1/fruit_tn2/loop_body/Tensordot/pfor/Reshape:output:0Msequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€Й
Gsequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : √
Bsequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2,sequential_1/fruit_tn2/pfor/Reshape:output:0Esequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1/shape:output:0Psequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ю
Csequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshapeKsequential_1/fruit_tn2/loop_body/Tensordot_1/transpose_1/pfor/Transpose:y:0Ksequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€Ї
>sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/ShapeShapeLsequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:Ц
Lsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ш
Nsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ш
Nsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:о
Fsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSliceGsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Usequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskШ
Nsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
Hsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSliceGsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
@sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/MinimumMinimumOsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0Qsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: В
@sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :щ
>sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/EqualEqualDsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Minimum:z:0Isequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: Ц
Asequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
Asequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"           
?sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/SelectSelectBsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Equal:z:0Jsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/t:output:0Jsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:Ш
Nsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
Hsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSliceGsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskШ
Nsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: Ъ
Psequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
Hsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSliceGsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskШ
Nsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
Hsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSliceGsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskн
>sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/stackPackQsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0Qsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0Qsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:ѓ
Bsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose	TransposeLsequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0Hsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€Т
@sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/ReshapeReshapeFsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose:y:0Gsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*+
_output_shapes
:€€€€€€€€€є
@sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape_1ShapeIsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:К
Hsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : І
>sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/splitSplitQsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split/split_dim:output:0Isequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_splitЛ
Hsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB Н
Jsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB М
Bsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1ReshapeGsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split:output:0Ssequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: Л
Hsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB Н
Jsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB М
Bsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2ReshapeGsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split:output:1Ssequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: Л
Hsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB Н
Jsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB М
Bsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3ReshapeGsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/split:output:2Ssequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ю
<sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/mulMulKsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0Ksequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: Н
Hsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePackKsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:0@sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:¶
Bsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4ReshapeIsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0Qsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€И
?sequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul=sequential_1/fruit_tn2/loop_body/Tensordot_1/Reshape:output:0Ksequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*(
_output_shapes
:М€€€€€€€€€Х
Jsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€н
Hsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePackSsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:0Ksequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:0Ksequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:≥
Bsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5ReshapeIsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0Qsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€Ю
Isequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ©
Dsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1	TransposeKsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0Rsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€М
=sequential_1/fruit_tn2/loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : •
8sequential_1/fruit_tn2/loop_body/Tensordot_1/pfor/concatConcatV2,sequential_1/fruit_tn2/pfor/Reshape:output:0;sequential_1/fruit_tn2/loop_body/Tensordot_1/shape:output:0Fsequential_1/fruit_tn2/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:М
9sequential_1/fruit_tn2/loop_body/Tensordot_1/pfor/ReshapeReshapeHsequential_1/fruit_tn2/loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:0Asequential_1/fruit_tn2/loop_body/Tensordot_1/pfor/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€ГГ
Asequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :€
?sequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/addAddV2Dsequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/perm:output:0Jsequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/add/y:output:0*
T0*
_output_shapes
:Х
Ksequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: Й
Gsequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : й
Bsequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concatConcatV2Tsequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/values_0:output:0Csequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/add:z:0Psequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ю
Esequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/Transpose	TransposeBsequential_1/fruit_tn2/loop_body/Tensordot_1/pfor/Reshape:output:0Ksequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€ГЙ
Gsequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : √
Bsequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concatConcatV2,sequential_1/fruit_tn2/pfor/Reshape:output:0Esequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1/shape:output:0Psequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Э
Csequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/ReshapeReshapeIsequential_1/fruit_tn2/loop_body/Tensordot_2/transpose/pfor/Transpose:y:0Ksequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€ГЇ
>sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/ShapeShapeLsequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:Ц
Lsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ш
Nsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ш
Nsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:о
Fsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_sliceStridedSliceGsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Usequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskШ
Nsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
Hsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1StridedSliceGsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
@sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/MinimumMinimumOsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice:output:0Qsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: В
@sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :щ
>sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/EqualEqualDsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Minimum:z:0Isequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Equal/y:output:0*
T0*
_output_shapes
: Ц
Asequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/tConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
Asequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/eConst*
_output_shapes
:*
dtype0*!
valueB"           
?sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/SelectSelectBsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Equal:z:0Jsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/t:output:0Jsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:Ш
Nsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
Hsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2StridedSliceGsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskШ
Nsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: Ъ
Psequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
Hsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3StridedSliceGsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskШ
Nsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
Psequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
Hsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4StridedSliceGsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape:output:0Wsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1:output:0Ysequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskн
>sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/stackPackQsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_2:output:0Qsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_3:output:0Qsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:ѓ
Bsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose	TransposeLsequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0Hsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€У
@sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/ReshapeReshapeFsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose:y:0Gsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:€€€€€€€€€Гє
@sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape_1ShapeIsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:К
Hsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : І
>sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/splitSplitQsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split/split_dim:output:0Isequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Shape_1:output:0*
T0*&
_output_shapes
:::*
	num_splitЛ
Hsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shapeConst*
_output_shapes
: *
dtype0*
valueB Н
Jsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1Const*
_output_shapes
: *
dtype0*
valueB М
Bsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1ReshapeGsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split:output:0Ssequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1/shape_1:output:0*
T0*
_output_shapes
: Л
Hsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shapeConst*
_output_shapes
: *
dtype0*
valueB Н
Jsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1Const*
_output_shapes
: *
dtype0*
valueB М
Bsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2ReshapeGsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split:output:1Ssequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: Л
Hsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shapeConst*
_output_shapes
: *
dtype0*
valueB Н
Jsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1Const*
_output_shapes
: *
dtype0*
valueB М
Bsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3ReshapeGsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/split:output:2Ssequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: ю
<sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/mulMulKsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0Ksequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: Н
Hsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shapePackKsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_1:output:0@sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:¶
Bsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4ReshapeIsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0Qsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€З
?sequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/MatMulMatMul=sequential_1/fruit_tn2/loop_body/Tensordot_2/Reshape:output:0Ksequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€Х
Jsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€н
Hsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shapePackSsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0:output:0Ksequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:0Ksequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:≥
Bsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5ReshapeIsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/MatMul:product:0Qsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€Ю
Isequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ©
Dsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1	TransposeKsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/Reshape_5:output:0Rsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Г
=sequential_1/fruit_tn2/loop_body/Tensordot_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : •
8sequential_1/fruit_tn2/loop_body/Tensordot_2/pfor/concatConcatV2,sequential_1/fruit_tn2/pfor/Reshape:output:0;sequential_1/fruit_tn2/loop_body/Tensordot_2/shape:output:0Fsequential_1/fruit_tn2/loop_body/Tensordot_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Д
9sequential_1/fruit_tn2/loop_body/Tensordot_2/pfor/ReshapeReshapeHsequential_1/fruit_tn2/loop_body/Tensordot_2/MatMul/pfor/transpose_1:y:0Asequential_1/fruit_tn2/loop_body/Tensordot_2/pfor/concat:output:0*
T0*(
_output_shapes
:€€€€€€€€€Гp
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
value	B :ћ
-sequential_1/fruit_tn2/loop_body/add/pfor/addAddV29sequential_1/fruit_tn2/loop_body/add/pfor/Rank_1:output:08sequential_1/fruit_tn2/loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: …
1sequential_1/fruit_tn2/loop_body/add/pfor/MaximumMaximum1sequential_1/fruit_tn2/loop_body/add/pfor/add:z:07sequential_1/fruit_tn2/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: °
/sequential_1/fruit_tn2/loop_body/add/pfor/ShapeShapeBsequential_1/fruit_tn2/loop_body/Tensordot_2/pfor/Reshape:output:0*
T0*
_output_shapes
:≈
-sequential_1/fruit_tn2/loop_body/add/pfor/subSub5sequential_1/fruit_tn2/loop_body/add/pfor/Maximum:z:07sequential_1/fruit_tn2/loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: Б
7sequential_1/fruit_tn2/loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:÷
1sequential_1/fruit_tn2/loop_body/add/pfor/ReshapeReshape1sequential_1/fruit_tn2/loop_body/add/pfor/sub:z:0@sequential_1/fruit_tn2/loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:~
4sequential_1/fruit_tn2/loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:‘
.sequential_1/fruit_tn2/loop_body/add/pfor/TileTile=sequential_1/fruit_tn2/loop_body/add/pfor/Tile/input:output:0:sequential_1/fruit_tn2/loop_body/add/pfor/Reshape:output:0*
T0*
_output_shapes
: З
=sequential_1/fruit_tn2/loop_body/add/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Й
?sequential_1/fruit_tn2/loop_body/add/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Й
?sequential_1/fruit_tn2/loop_body/add/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
7sequential_1/fruit_tn2/loop_body/add/pfor/strided_sliceStridedSlice8sequential_1/fruit_tn2/loop_body/add/pfor/Shape:output:0Fsequential_1/fruit_tn2/loop_body/add/pfor/strided_slice/stack:output:0Hsequential_1/fruit_tn2/loop_body/add/pfor/strided_slice/stack_1:output:0Hsequential_1/fruit_tn2/loop_body/add/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskЙ
?sequential_1/fruit_tn2/loop_body/add/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Л
Asequential_1/fruit_tn2/loop_body/add/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Л
Asequential_1/fruit_tn2/loop_body/add/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:І
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
value	B : й
0sequential_1/fruit_tn2/loop_body/add/pfor/concatConcatV2@sequential_1/fruit_tn2/loop_body/add/pfor/strided_slice:output:07sequential_1/fruit_tn2/loop_body/add/pfor/Tile:output:0Bsequential_1/fruit_tn2/loop_body/add/pfor/strided_slice_1:output:0>sequential_1/fruit_tn2/loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:р
3sequential_1/fruit_tn2/loop_body/add/pfor/Reshape_1ReshapeBsequential_1/fruit_tn2/loop_body/Tensordot_2/pfor/Reshape:output:09sequential_1/fruit_tn2/loop_body/add/pfor/concat:output:0*
T0*(
_output_shapes
:€€€€€€€€€Гж
/sequential_1/fruit_tn2/loop_body/add/pfor/AddV2AddV2<sequential_1/fruit_tn2/loop_body/add/pfor/Reshape_1:output:0;sequential_1/fruit_tn2/loop_body/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Гu
$sequential_1/fruit_tn2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Г   ј
sequential_1/fruit_tn2/ReshapeReshape3sequential_1/fruit_tn2/loop_body/add/pfor/AddV2:z:0-sequential_1/fruit_tn2/Reshape/shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ГЕ
sequential_1/fruit_tn2/SoftmaxSoftmax'sequential_1/fruit_tn2/Reshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ГШ
1sequential_1/fruit_tn2/ActivityRegularizer/SquareSquare(sequential_1/fruit_tn2/Softmax:softmax:0*
T0*(
_output_shapes
:€€€€€€€€€ГБ
0sequential_1/fruit_tn2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       »
.sequential_1/fruit_tn2/ActivityRegularizer/SumSum5sequential_1/fruit_tn2/ActivityRegularizer/Square:y:09sequential_1/fruit_tn2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: u
0sequential_1/fruit_tn2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ђ≈'7 
.sequential_1/fruit_tn2/ActivityRegularizer/mulMul9sequential_1/fruit_tn2/ActivityRegularizer/mul/x:output:07sequential_1/fruit_tn2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: И
0sequential_1/fruit_tn2/ActivityRegularizer/ShapeShape(sequential_1/fruit_tn2/Softmax:softmax:0*
T0*
_output_shapes
:И
>sequential_1/fruit_tn2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: К
@sequential_1/fruit_tn2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:К
@sequential_1/fruit_tn2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
8sequential_1/fruit_tn2/ActivityRegularizer/strided_sliceStridedSlice9sequential_1/fruit_tn2/ActivityRegularizer/Shape:output:0Gsequential_1/fruit_tn2/ActivityRegularizer/strided_slice/stack:output:0Isequential_1/fruit_tn2/ActivityRegularizer/strided_slice/stack_1:output:0Isequential_1/fruit_tn2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask™
/sequential_1/fruit_tn2/ActivityRegularizer/CastCastAsequential_1/fruit_tn2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: «
2sequential_1/fruit_tn2/ActivityRegularizer/truedivRealDiv2sequential_1/fruit_tn2/ActivityRegularizer/mul:z:03sequential_1/fruit_tn2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: x
IdentityIdentity(sequential_1/fruit_tn2/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Г•
NoOpNoOp+^sequential_1/conv2d/BiasAdd/ReadVariableOp*^sequential_1/conv2d/Conv2D/ReadVariableOp&^sequential_1/conv2dmpo/ReadVariableOp(^sequential_1/conv2dmpo/ReadVariableOp_10^sequential_1/conv2dmpo/Reshape_2/ReadVariableOp(^sequential_1/conv2dmpo_1/ReadVariableOp*^sequential_1/conv2dmpo_1/ReadVariableOp_1*^sequential_1/conv2dmpo_1/ReadVariableOp_2*^sequential_1/conv2dmpo_1/ReadVariableOp_32^sequential_1/conv2dmpo_1/Reshape_2/ReadVariableOp0^sequential_1/fruit_tn1/loop_body/ReadVariableOp2^sequential_1/fruit_tn1/loop_body/ReadVariableOp_12^sequential_1/fruit_tn1/loop_body/ReadVariableOp_24^sequential_1/fruit_tn1/loop_body/add/ReadVariableOp0^sequential_1/fruit_tn2/loop_body/ReadVariableOp2^sequential_1/fruit_tn2/loop_body/ReadVariableOp_12^sequential_1/fruit_tn2/loop_body/ReadVariableOp_24^sequential_1/fruit_tn2/loop_body/add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€22: : : : : : : : : : : : : : : : : : 2X
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
:€€€€€€€€€22
"
_user_specified_name
input_12
їз
Ї
E__inference_fruit_tn2_layer_call_and_return_conditional_losses_413174

inputs3
!loop_body_readvariableop_resource:>
#loop_body_readvariableop_1_resource:Г5
#loop_body_readvariableop_2_resource:4
%loop_body_add_readvariableop_resource:	Г
identityИҐloop_body/ReadVariableOpҐloop_body/ReadVariableOp_1Ґloop_body/ReadVariableOp_2Ґloop_body/add/ReadVariableOp;
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
valueB:—
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
value	B : Э
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
valueB:Г
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
value	B : †
loop_body/SelectV2SelectV2loop_body/Greater:z:0)loop_body/PlaceholderWithDefault:output:0loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: Y
loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ≠
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
valueB"         И
loop_body/ReshapeReshapeloop_body/GatherV2:output:0 loop_body/Reshape/shape:output:0*
T0*"
_output_shapes
:z
loop_body/ReadVariableOpReadVariableOp!loop_body_readvariableop_resource*
_output_shapes

:*
dtype0З
loop_body/ReadVariableOp_1ReadVariableOp#loop_body_readvariableop_1_resource*'
_output_shapes
:Г*
dtype0~
loop_body/ReadVariableOp_2ReadVariableOp#loop_body_readvariableop_2_resource*
_output_shapes

:*
dtype0w
"loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          †
loop_body/Tensordot/transpose	Transposeloop_body/Reshape:output:0+loop_body/Tensordot/transpose/perm:output:0*
T0*"
_output_shapes
:r
!loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ю
loop_body/Tensordot/ReshapeReshape!loop_body/Tensordot/transpose:y:0*loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:Х
loop_body/Tensordot/MatMulMatMul$loop_body/Tensordot/Reshape:output:0 loop_body/ReadVariableOp:value:0*
T0*
_output_shapes

:n
loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         Х
loop_body/TensordotReshape$loop_body/Tensordot/MatMul:product:0"loop_body/Tensordot/shape:output:0*
T0*"
_output_shapes
:}
$loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ±
loop_body/Tensordot_1/transpose	Transpose"loop_body/ReadVariableOp_1:value:0-loop_body/Tensordot_1/transpose/perm:output:0*
T0*'
_output_shapes
:Гt
#loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"     •
loop_body/Tensordot_1/ReshapeReshape#loop_body/Tensordot_1/transpose:y:0,loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes
:	М{
&loop_body/Tensordot_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ™
!loop_body/Tensordot_1/transpose_1	Transposeloop_body/Tensordot:output:0/loop_body/Tensordot_1/transpose_1/perm:output:0*
T0*"
_output_shapes
:v
%loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ™
loop_body/Tensordot_1/Reshape_1Reshape%loop_body/Tensordot_1/transpose_1:y:0.loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:Ґ
loop_body/Tensordot_1/MatMulMatMul&loop_body/Tensordot_1/Reshape:output:0(loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	Мp
loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"Г         Ь
loop_body/Tensordot_1Reshape&loop_body/Tensordot_1/MatMul:product:0$loop_body/Tensordot_1/shape:output:0*
T0*#
_output_shapes
:Гt
#loop_body/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      £
loop_body/Tensordot_2/ReshapeReshape"loop_body/ReadVariableOp_2:value:0,loop_body/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes

:y
$loop_body/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ©
loop_body/Tensordot_2/transpose	Transposeloop_body/Tensordot_1:output:0-loop_body/Tensordot_2/transpose/perm:output:0*
T0*#
_output_shapes
:Гv
%loop_body/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   Г   ©
loop_body/Tensordot_2/Reshape_1Reshape#loop_body/Tensordot_2/transpose:y:0.loop_body/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	ГҐ
loop_body/Tensordot_2/MatMulMatMul&loop_body/Tensordot_2/Reshape:output:0(loop_body/Tensordot_2/Reshape_1:output:0*
T0*
_output_shapes
:	Гf
loop_body/Tensordot_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:ГФ
loop_body/Tensordot_2Reshape&loop_body/Tensordot_2/MatMul:product:0$loop_body/Tensordot_2/shape:output:0*
T0*
_output_shapes	
:Г
loop_body/add/ReadVariableOpReadVariableOp%loop_body_add_readvariableop_resource*
_output_shapes	
:Г*
dtype0В
loop_body/addAddV2loop_body/Tensordot_2:output:0$loop_body/add/ReadVariableOp:value:0*
T0*
_output_shapes	
:Г\
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
:€€€€€€€€€^
loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : _
loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ф
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
value	B :Ъ
loop_body/SelectV2/pfor/add_1AddV2'loop_body/SelectV2/pfor/Rank_2:output:0(loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: Х
loop_body/SelectV2/pfor/MaximumMaximum'loop_body/SelectV2/pfor/Rank_1:output:0loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: Х
!loop_body/SelectV2/pfor/Maximum_1Maximum!loop_body/SelectV2/pfor/add_1:z:0#loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: `
loop_body/SelectV2/pfor/ShapeShapepfor/range:output:0*
T0*
_output_shapes
:У
loop_body/SelectV2/pfor/subSub%loop_body/SelectV2/pfor/Maximum_1:z:0'loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: o
%loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:†
loop_body/SelectV2/pfor/ReshapeReshapeloop_body/SelectV2/pfor/sub:z:0.loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:l
"loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Ю
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
valueB:«
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
valueB:Ћ
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
value	B : П
loop_body/SelectV2/pfor/concatConcatV2.loop_body/SelectV2/pfor/strided_slice:output:0%loop_body/SelectV2/pfor/Tile:output:00loop_body/SelectV2/pfor/strided_slice_1:output:0,loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ш
!loop_body/SelectV2/pfor/Reshape_1Reshapepfor/range:output:0'loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:€€€€€€€€€Љ
 loop_body/SelectV2/pfor/SelectV2SelectV2loop_body/Greater:z:0*loop_body/SelectV2/pfor/Reshape_1:output:0loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:€€€€€€€€€g
%loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : д
 loop_body/GatherV2/pfor/GatherV2GatherV2inputs)loop_body/SelectV2/pfor/SelectV2:output:0.loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:€€€€€€€€€@d
"loop_body/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : љ
loop_body/Reshape/pfor/concatConcatV2pfor/Reshape:output:0 loop_body/Reshape/shape:output:0+loop_body/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ґ
loop_body/Reshape/pfor/ReshapeReshape)loop_body/GatherV2/pfor/GatherV2:output:0&loop_body/Reshape/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€j
(loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :і
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
value	B : Е
)loop_body/Tensordot/transpose/pfor/concatConcatV2;loop_body/Tensordot/transpose/pfor/concat/values_0:output:0*loop_body/Tensordot/transpose/pfor/add:z:07loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:–
,loop_body/Tensordot/transpose/pfor/Transpose	Transpose'loop_body/Reshape/pfor/Reshape:output:02loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€n
,loop_body/Tensordot/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : џ
'loop_body/Tensordot/Reshape/pfor/concatConcatV2pfor/Reshape:output:0*loop_body/Tensordot/Reshape/shape:output:05loop_body/Tensordot/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ќ
(loop_body/Tensordot/Reshape/pfor/ReshapeReshape0loop_body/Tensordot/transpose/pfor/Transpose:y:00loop_body/Tensordot/Reshape/pfor/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€Ж
%loop_body/Tensordot/MatMul/pfor/ShapeShape1loop_body/Tensordot/Reshape/pfor/Reshape:output:0*
T0*
_output_shapes
:q
/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Џ
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
valueB љ
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
valueB Ѕ
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
valueB Ѕ
)loop_body/Tensordot/MatMul/pfor/Reshape_2Reshape.loop_body/Tensordot/MatMul/pfor/split:output:2:loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ±
#loop_body/Tensordot/MatMul/pfor/mulMul0loop_body/Tensordot/MatMul/pfor/Reshape:output:02loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: ¬
/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack'loop_body/Tensordot/MatMul/pfor/mul:z:02loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:№
)loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape1loop_body/Tensordot/Reshape/pfor/Reshape:output:08loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€Є
&loop_body/Tensordot/MatMul/pfor/MatMulMatMul2loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0 loop_body/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€|
1loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€З
/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePack0loop_body/Tensordot/MatMul/pfor/Reshape:output:02loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0:loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:÷
)loop_body/Tensordot/MatMul/pfor/Reshape_4Reshape0loop_body/Tensordot/MatMul/pfor/MatMul:product:08loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€f
$loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : √
loop_body/Tensordot/pfor/concatConcatV2pfor/Reshape:output:0"loop_body/Tensordot/shape:output:0-loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:√
 loop_body/Tensordot/pfor/ReshapeReshape2loop_body/Tensordot/MatMul/pfor/Reshape_4:output:0(loop_body/Tensordot/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€n
,loop_body/Tensordot_1/transpose_1/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :ј
*loop_body/Tensordot_1/transpose_1/pfor/addAddV2/loop_body/Tensordot_1/transpose_1/perm:output:05loop_body/Tensordot_1/transpose_1/pfor/add/y:output:0*
T0*
_output_shapes
:А
6loop_body/Tensordot_1/transpose_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: t
2loop_body/Tensordot_1/transpose_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Х
-loop_body/Tensordot_1/transpose_1/pfor/concatConcatV2?loop_body/Tensordot_1/transpose_1/pfor/concat/values_0:output:0.loop_body/Tensordot_1/transpose_1/pfor/add:z:0;loop_body/Tensordot_1/transpose_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Џ
0loop_body/Tensordot_1/transpose_1/pfor/Transpose	Transpose)loop_body/Tensordot/pfor/Reshape:output:06loop_body/Tensordot_1/transpose_1/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€r
0loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : з
+loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2pfor/Reshape:output:0.loop_body/Tensordot_1/Reshape_1/shape:output:09loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ў
,loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape4loop_body/Tensordot_1/transpose_1/pfor/Transpose:y:04loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€М
'loop_body/Tensordot_1/MatMul/pfor/ShapeShape5loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
5loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0>loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЋ
)loop_body/Tensordot_1/MatMul/pfor/MinimumMinimum8loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: k
)loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :і
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
valueB"          о
(loop_body/Tensordot_1/MatMul/pfor/SelectSelect+loop_body/Tensordot_1/MatMul/pfor/Equal:z:03loop_body/Tensordot_1/MatMul/pfor/Select/t:output:03loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:Б
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskС
'loop_body/Tensordot_1/MatMul/pfor/stackPack:loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:к
+loop_body/Tensordot_1/MatMul/pfor/transpose	Transpose5loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:01loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€Ќ
)loop_body/Tensordot_1/MatMul/pfor/ReshapeReshape/loop_body/Tensordot_1/MatMul/pfor/transpose:y:00loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*+
_output_shapes
:€€€€€€€€€Л
)loop_body/Tensordot_1/MatMul/pfor/Shape_1Shape2loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:s
1loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : в
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
valueB «
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
valueB «
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
valueB «
+loop_body/Tensordot_1/MatMul/pfor/Reshape_3Reshape0loop_body/Tensordot_1/MatMul/pfor/split:output:2<loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: є
%loop_body/Tensordot_1/MatMul/pfor/mulMul4loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: »
1loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePack4loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:0)loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:б
+loop_body/Tensordot_1/MatMul/pfor/Reshape_4Reshape2loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0:loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€√
(loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul&loop_body/Tensordot_1/Reshape:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*(
_output_shapes
:М€€€€€€€€€~
3loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€С
1loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePack<loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:о
+loop_body/Tensordot_1/MatMul/pfor/Reshape_5Reshape2loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0:loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€З
2loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          д
-loop_body/Tensordot_1/MatMul/pfor/transpose_1	Transpose4loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0;loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Мh
&loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : …
!loop_body/Tensordot_1/pfor/concatConcatV2pfor/Reshape:output:0$loop_body/Tensordot_1/shape:output:0/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:«
"loop_body/Tensordot_1/pfor/ReshapeReshape1loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:0*loop_body/Tensordot_1/pfor/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€Гl
*loop_body/Tensordot_2/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ї
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
value	B : Н
+loop_body/Tensordot_2/transpose/pfor/concatConcatV2=loop_body/Tensordot_2/transpose/pfor/concat/values_0:output:0,loop_body/Tensordot_2/transpose/pfor/add:z:09loop_body/Tensordot_2/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ў
.loop_body/Tensordot_2/transpose/pfor/Transpose	Transpose+loop_body/Tensordot_1/pfor/Reshape:output:04loop_body/Tensordot_2/transpose/pfor/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€Гr
0loop_body/Tensordot_2/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : з
+loop_body/Tensordot_2/Reshape_1/pfor/concatConcatV2pfor/Reshape:output:0.loop_body/Tensordot_2/Reshape_1/shape:output:09loop_body/Tensordot_2/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ў
,loop_body/Tensordot_2/Reshape_1/pfor/ReshapeReshape2loop_body/Tensordot_2/transpose/pfor/Transpose:y:04loop_body/Tensordot_2/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€ГМ
'loop_body/Tensordot_2/MatMul/pfor/ShapeShape5loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
5loop_body/Tensordot_2/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
/loop_body/Tensordot_2/MatMul/pfor/strided_sliceStridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0>loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_1StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЋ
)loop_body/Tensordot_2/MatMul/pfor/MinimumMinimum8loop_body/Tensordot_2/MatMul/pfor/strided_slice:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: k
)loop_body/Tensordot_2/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :і
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
valueB"          о
(loop_body/Tensordot_2/MatMul/pfor/SelectSelect+loop_body/Tensordot_2/MatMul/pfor/Equal:z:03loop_body/Tensordot_2/MatMul/pfor/Select/t:output:03loop_body/Tensordot_2/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:Б
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_2StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_3StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_4StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskС
'loop_body/Tensordot_2/MatMul/pfor/stackPack:loop_body/Tensordot_2/MatMul/pfor/strided_slice_2:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_3:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:к
+loop_body/Tensordot_2/MatMul/pfor/transpose	Transpose5loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:01loop_body/Tensordot_2/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€ќ
)loop_body/Tensordot_2/MatMul/pfor/ReshapeReshape/loop_body/Tensordot_2/MatMul/pfor/transpose:y:00loop_body/Tensordot_2/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:€€€€€€€€€ГЛ
)loop_body/Tensordot_2/MatMul/pfor/Shape_1Shape2loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:s
1loop_body/Tensordot_2/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : в
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
valueB «
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
valueB «
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
valueB «
+loop_body/Tensordot_2/MatMul/pfor/Reshape_3Reshape0loop_body/Tensordot_2/MatMul/pfor/split:output:2<loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: є
%loop_body/Tensordot_2/MatMul/pfor/mulMul4loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: »
1loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shapePack4loop_body/Tensordot_2/MatMul/pfor/Reshape_1:output:0)loop_body/Tensordot_2/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:б
+loop_body/Tensordot_2/MatMul/pfor/Reshape_4Reshape2loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0:loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€¬
(loop_body/Tensordot_2/MatMul/pfor/MatMulMatMul&loop_body/Tensordot_2/Reshape:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€~
3loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€С
1loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shapePack<loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:о
+loop_body/Tensordot_2/MatMul/pfor/Reshape_5Reshape2loop_body/Tensordot_2/MatMul/pfor/MatMul:product:0:loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€З
2loop_body/Tensordot_2/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          д
-loop_body/Tensordot_2/MatMul/pfor/transpose_1	Transpose4loop_body/Tensordot_2/MatMul/pfor/Reshape_5:output:0;loop_body/Tensordot_2/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Гh
&loop_body/Tensordot_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : …
!loop_body/Tensordot_2/pfor/concatConcatV2pfor/Reshape:output:0$loop_body/Tensordot_2/shape:output:0/loop_body/Tensordot_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:њ
"loop_body/Tensordot_2/pfor/ReshapeReshape1loop_body/Tensordot_2/MatMul/pfor/transpose_1:y:0*loop_body/Tensordot_2/pfor/concat:output:0*
T0*(
_output_shapes
:€€€€€€€€€ГY
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
value	B :З
loop_body/add/pfor/addAddV2"loop_body/add/pfor/Rank_1:output:0!loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: Д
loop_body/add/pfor/MaximumMaximumloop_body/add/pfor/add:z:0 loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: s
loop_body/add/pfor/ShapeShape+loop_body/Tensordot_2/pfor/Reshape:output:0*
T0*
_output_shapes
:А
loop_body/add/pfor/subSubloop_body/add/pfor/Maximum:z:0 loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: j
 loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:С
loop_body/add/pfor/ReshapeReshapeloop_body/add/pfor/sub:z:0)loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:g
loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:П
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
valueB:Ѓ
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
valueB:і
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
value	B : ц
loop_body/add/pfor/concatConcatV2)loop_body/add/pfor/strided_slice:output:0 loop_body/add/pfor/Tile:output:0+loop_body/add/pfor/strided_slice_1:output:0'loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ђ
loop_body/add/pfor/Reshape_1Reshape+loop_body/Tensordot_2/pfor/Reshape:output:0"loop_body/add/pfor/concat:output:0*
T0*(
_output_shapes
:€€€€€€€€€Г°
loop_body/add/pfor/AddV2AddV2%loop_body/add/pfor/Reshape_1:output:0$loop_body/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Г^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€Г   {
ReshapeReshapeloop_body/add/pfor/AddV2:z:0Reshape/shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ГW
SoftmaxSoftmaxReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€Гa
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ГЇ
NoOpNoOp^loop_body/ReadVariableOp^loop_body/ReadVariableOp_1^loop_body/ReadVariableOp_2^loop_body/add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 24
loop_body/ReadVariableOploop_body/ReadVariableOp28
loop_body/ReadVariableOp_1loop_body/ReadVariableOp_128
loop_body/ReadVariableOp_2loop_body/ReadVariableOp_22<
loop_body/add/ReadVariableOploop_body/add/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
К
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
:€€€€€€€€€G
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
 *ђ≈'7I
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
у_
”
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
conv2dmpo_1_413728:	А&
fruit_tn1_413740:+
fruit_tn1_413742:А&
fruit_tn1_413744:&
fruit_tn1_413746:"
fruit_tn2_413758:+
fruit_tn2_413760:Г"
fruit_tn2_413762:
fruit_tn2_413764:	Г
identity

identity_1

identity_2

identity_3

identity_4ИҐconv2d/StatefulPartitionedCallҐ!conv2dmpo/StatefulPartitionedCallҐ#conv2dmpo_1/StatefulPartitionedCallҐ dropout1/StatefulPartitionedCallҐ!fruit_tn1/StatefulPartitionedCallҐ!fruit_tn2/StatefulPartitionedCallх
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_12conv2d_413699conv2d_413701*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€//*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_412370і
!conv2dmpo/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2dmpo_413704conv2dmpo_413706conv2dmpo_413708*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€//*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2dmpo_layer_call_and_return_conditional_losses_412438–
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
GPU2*0J 8В *:
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
valueB:з
+conv2dmpo/ActivityRegularizer/strided_sliceStridedSlice,conv2dmpo/ActivityRegularizer/Shape:output:0:conv2dmpo/ActivityRegularizer/strided_slice/stack:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_1:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
"conv2dmpo/ActivityRegularizer/CastCast4conv2dmpo/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ±
%conv2dmpo/ActivityRegularizer/truedivRealDiv6conv2dmpo/ActivityRegularizer/PartitionedCall:output:0&conv2dmpo/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: с
max_pooling2d/PartitionedCallPartitionedCall*conv2dmpo/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_412299к
#conv2dmpo_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2dmpo_1_413720conv2dmpo_1_413722conv2dmpo_1_413724conv2dmpo_1_413726conv2dmpo_1_413728*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2dmpo_1_layer_call_and_return_conditional_losses_412543÷
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
GPU2*0J 8В *<
f7R5
3__inference_conv2dmpo_1_activity_regularizer_412315Б
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
valueB:с
-conv2dmpo_1/ActivityRegularizer/strided_sliceStridedSlice.conv2dmpo_1/ActivityRegularizer/Shape:output:0<conv2dmpo_1/ActivityRegularizer/strided_slice/stack:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_1:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskФ
$conv2dmpo_1/ActivityRegularizer/CastCast6conv2dmpo_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ј
'conv2dmpo_1/ActivityRegularizer/truedivRealDiv8conv2dmpo_1/ActivityRegularizer/PartitionedCall:output:0(conv2dmpo_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ш
max_pooling2d_1/PartitionedCallPartitionedCall,conv2dmpo_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_412324Ѕ
!fruit_tn1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0fruit_tn1_413740fruit_tn1_413742fruit_tn1_413744fruit_tn1_413746*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_fruit_tn1_layer_call_and_return_conditional_losses_412856–
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
GPU2*0J 8В *:
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
valueB:з
+fruit_tn1/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn1/ActivityRegularizer/Shape:output:0:fruit_tn1/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
"fruit_tn1/ActivityRegularizer/CastCast4fruit_tn1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ±
%fruit_tn1/ActivityRegularizer/truedivRealDiv6fruit_tn1/ActivityRegularizer/PartitionedCall:output:0&fruit_tn1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: п
 dropout1/StatefulPartitionedCallStatefulPartitionedCall*fruit_tn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout1_layer_call_and_return_conditional_losses_413290√
!fruit_tn2/StatefulPartitionedCallStatefulPartitionedCall)dropout1/StatefulPartitionedCall:output:0fruit_tn2_413758fruit_tn2_413760fruit_tn2_413762fruit_tn2_413764*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Г*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_fruit_tn2_layer_call_and_return_conditional_losses_413174–
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
GPU2*0J 8В *:
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
valueB:з
+fruit_tn2/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn2/ActivityRegularizer/Shape:output:0:fruit_tn2/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
"fruit_tn2/ActivityRegularizer/CastCast4fruit_tn2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ±
%fruit_tn2/ActivityRegularizer/truedivRealDiv6fruit_tn2/ActivityRegularizer/PartitionedCall:output:0&fruit_tn2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: z
IdentityIdentity*fruit_tn2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Гi

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
: Ь
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
?:€€€€€€€€€22: : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2F
!conv2dmpo/StatefulPartitionedCall!conv2dmpo/StatefulPartitionedCall2J
#conv2dmpo_1/StatefulPartitionedCall#conv2dmpo_1/StatefulPartitionedCall2D
 dropout1/StatefulPartitionedCall dropout1/StatefulPartitionedCall2F
!fruit_tn1/StatefulPartitionedCall!fruit_tn1/StatefulPartitionedCall2F
!fruit_tn2/StatefulPartitionedCall!fruit_tn2/StatefulPartitionedCall:Y U
/
_output_shapes
:€€€€€€€€€22
"
_user_specified_name
input_12
й
Ь
'__inference_conv2d_layer_call_fn_415549

inputs!
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€//*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_412370w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€//`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€22: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs
ѕ
ґ
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
	unknown_8:	А
	unknown_9:%

unknown_10:А 

unknown_11: 

unknown_12:

unknown_13:%

unknown_14:Г

unknown_15:

unknown_16:	Г
identityИҐStatefulPartitionedCallУ
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
:€€€€€€€€€Г*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_412277p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Г`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€22: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:€€€€€€€€€22
"
_user_specified_name
input_12
£°
Љ 
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

identity_1ИҐMergeV2Checkpointsw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ф0
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*Э0
valueУ0BР0QB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/end_node_first/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-1/end_node_last/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/end_node_first/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-2/middle_node_0/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-2/middle_node_1/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-2/end_node_last/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/a_var/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/b_var/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/c_var/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/a_var/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/b_var/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/c_var/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/end_node_first/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/end_node_last/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/end_node_first/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/middle_node_0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/middle_node_1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/end_node_last/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/a_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/b_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/c_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/a_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/b_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/c_var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/end_node_first/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/end_node_last/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/end_node_first/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/middle_node_0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/middle_node_1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/end_node_last/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/a_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/b_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/c_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/a_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/b_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/c_var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-1/end_node_first/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/end_node_last/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-2/end_node_first/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/middle_node_0/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/middle_node_1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/end_node_last/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/a_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/b_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/c_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/a_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/b_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/c_var/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHТ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*Ј
value≠B™QB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Р
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop3savev2_conv2dmpo_end_node_first_read_readvariableop2savev2_conv2dmpo_end_node_last_read_readvariableop)savev2_conv2dmpo_bias_read_readvariableop5savev2_conv2dmpo_1_end_node_first_read_readvariableop4savev2_conv2dmpo_1_middle_node_0_read_readvariableop4savev2_conv2dmpo_1_middle_node_1_read_readvariableop4savev2_conv2dmpo_1_end_node_last_read_readvariableop+savev2_conv2dmpo_1_bias_read_readvariableopsavev2_a_read_readvariableopsavev2_b_read_readvariableopsavev2_c_read_readvariableopsavev2_bias_read_readvariableopsavev2_a_1_read_readvariableopsavev2_b_1_read_readvariableopsavev2_c_1_read_readvariableop!savev2_bias_1_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop:savev2_adam_conv2dmpo_end_node_first_m_read_readvariableop9savev2_adam_conv2dmpo_end_node_last_m_read_readvariableop0savev2_adam_conv2dmpo_bias_m_read_readvariableop<savev2_adam_conv2dmpo_1_end_node_first_m_read_readvariableop;savev2_adam_conv2dmpo_1_middle_node_0_m_read_readvariableop;savev2_adam_conv2dmpo_1_middle_node_1_m_read_readvariableop;savev2_adam_conv2dmpo_1_end_node_last_m_read_readvariableop2savev2_adam_conv2dmpo_1_bias_m_read_readvariableop#savev2_adam_a_m_read_readvariableop#savev2_adam_b_m_read_readvariableop#savev2_adam_c_m_read_readvariableop&savev2_adam_bias_m_read_readvariableop%savev2_adam_a_m_1_read_readvariableop%savev2_adam_b_m_1_read_readvariableop%savev2_adam_c_m_1_read_readvariableop(savev2_adam_bias_m_1_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop:savev2_adam_conv2dmpo_end_node_first_v_read_readvariableop9savev2_adam_conv2dmpo_end_node_last_v_read_readvariableop0savev2_adam_conv2dmpo_bias_v_read_readvariableop<savev2_adam_conv2dmpo_1_end_node_first_v_read_readvariableop;savev2_adam_conv2dmpo_1_middle_node_0_v_read_readvariableop;savev2_adam_conv2dmpo_1_middle_node_1_v_read_readvariableop;savev2_adam_conv2dmpo_1_end_node_last_v_read_readvariableop2savev2_adam_conv2dmpo_1_bias_v_read_readvariableop#savev2_adam_a_v_read_readvariableop#savev2_adam_b_v_read_readvariableop#savev2_adam_c_v_read_readvariableop&savev2_adam_bias_v_read_readvariableop%savev2_adam_a_v_1_read_readvariableop%savev2_adam_b_v_1_read_readvariableop%savev2_adam_c_v_1_read_readvariableop(savev2_adam_bias_v_1_read_readvariableop2savev2_adam_conv2d_kernel_vhat_read_readvariableop0savev2_adam_conv2d_bias_vhat_read_readvariableop=savev2_adam_conv2dmpo_end_node_first_vhat_read_readvariableop<savev2_adam_conv2dmpo_end_node_last_vhat_read_readvariableop3savev2_adam_conv2dmpo_bias_vhat_read_readvariableop?savev2_adam_conv2dmpo_1_end_node_first_vhat_read_readvariableop>savev2_adam_conv2dmpo_1_middle_node_0_vhat_read_readvariableop>savev2_adam_conv2dmpo_1_middle_node_1_vhat_read_readvariableop>savev2_adam_conv2dmpo_1_end_node_last_vhat_read_readvariableop5savev2_adam_conv2dmpo_1_bias_vhat_read_readvariableop&savev2_adam_a_vhat_read_readvariableop&savev2_adam_b_vhat_read_readvariableop&savev2_adam_c_vhat_read_readvariableop)savev2_adam_bias_vhat_read_readvariableop(savev2_adam_a_vhat_1_read_readvariableop(savev2_adam_b_vhat_1_read_readvariableop(savev2_adam_c_vhat_1_read_readvariableop+savev2_adam_bias_vhat_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *_
dtypesU
S2Q	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
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

identity_1Identity_1:output:0*°
_input_shapesП
М: ::::::::::А::А::::Г::Г: : : : : : ::::::::::::А::А::::Г::Г::::::::::А::А::::Г::Г::::::::::А::А::::Г::Г: 2(
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
:А:($
"
_output_shapes
::-)
'
_output_shapes
:А:($
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
:Г:$ 

_output_shapes

::!

_output_shapes	
:Г:
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
:А:(%$
"
_output_shapes
::-&)
'
_output_shapes
:А:('$
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
:Г:$+ 

_output_shapes

::!,

_output_shapes	
:Г:,-(
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
:А:(7$
"
_output_shapes
::-8)
'
_output_shapes
:А:(9$
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
:Г:$= 

_output_shapes

::!>

_output_shapes	
:Г:,?(
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
:А:(I$
"
_output_shapes
::-J)
'
_output_shapes
:А:(K$
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
:Г:$O 

_output_shapes

::!P

_output_shapes	
:Г:Q

_output_shapes
: 
Е
љ
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
	unknown_8:	А
	unknown_9:%

unknown_10:А 

unknown_11: 

unknown_12:

unknown_13:%

unknown_14:Г

unknown_15:

unknown_16:	Г
identityИҐStatefulPartitionedCallƒ
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
:€€€€€€€€€Г: : : : *4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_413524p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Г`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€22: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs
…^
Ѓ
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
conv2dmpo_1_412552:	А&
fruit_tn1_412857:+
fruit_tn1_412859:А&
fruit_tn1_412861:&
fruit_tn1_412863:"
fruit_tn2_413175:+
fruit_tn2_413177:Г"
fruit_tn2_413179:
fruit_tn2_413181:	Г
identity

identity_1

identity_2

identity_3

identity_4ИҐconv2d/StatefulPartitionedCallҐ!conv2dmpo/StatefulPartitionedCallҐ#conv2dmpo_1/StatefulPartitionedCallҐ!fruit_tn1/StatefulPartitionedCallҐ!fruit_tn2/StatefulPartitionedCallу
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_412371conv2d_412373*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€//*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_412370і
!conv2dmpo/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2dmpo_412439conv2dmpo_412441conv2dmpo_412443*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€//*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2dmpo_layer_call_and_return_conditional_losses_412438–
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
GPU2*0J 8В *:
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
valueB:з
+conv2dmpo/ActivityRegularizer/strided_sliceStridedSlice,conv2dmpo/ActivityRegularizer/Shape:output:0:conv2dmpo/ActivityRegularizer/strided_slice/stack:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_1:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
"conv2dmpo/ActivityRegularizer/CastCast4conv2dmpo/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ±
%conv2dmpo/ActivityRegularizer/truedivRealDiv6conv2dmpo/ActivityRegularizer/PartitionedCall:output:0&conv2dmpo/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: с
max_pooling2d/PartitionedCallPartitionedCall*conv2dmpo/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_412299к
#conv2dmpo_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2dmpo_1_412544conv2dmpo_1_412546conv2dmpo_1_412548conv2dmpo_1_412550conv2dmpo_1_412552*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2dmpo_1_layer_call_and_return_conditional_losses_412543÷
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
GPU2*0J 8В *<
f7R5
3__inference_conv2dmpo_1_activity_regularizer_412315Б
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
valueB:с
-conv2dmpo_1/ActivityRegularizer/strided_sliceStridedSlice.conv2dmpo_1/ActivityRegularizer/Shape:output:0<conv2dmpo_1/ActivityRegularizer/strided_slice/stack:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_1:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskФ
$conv2dmpo_1/ActivityRegularizer/CastCast6conv2dmpo_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ј
'conv2dmpo_1/ActivityRegularizer/truedivRealDiv8conv2dmpo_1/ActivityRegularizer/PartitionedCall:output:0(conv2dmpo_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ш
max_pooling2d_1/PartitionedCallPartitionedCall,conv2dmpo_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_412324Ѕ
!fruit_tn1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0fruit_tn1_412857fruit_tn1_412859fruit_tn1_412861fruit_tn1_412863*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_fruit_tn1_layer_call_and_return_conditional_losses_412856–
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
GPU2*0J 8В *:
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
valueB:з
+fruit_tn1/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn1/ActivityRegularizer/Shape:output:0:fruit_tn1/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
"fruit_tn1/ActivityRegularizer/CastCast4fruit_tn1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ±
%fruit_tn1/ActivityRegularizer/truedivRealDiv6fruit_tn1/ActivityRegularizer/PartitionedCall:output:0&fruit_tn1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: я
dropout1/PartitionedCallPartitionedCall*fruit_tn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout1_layer_call_and_return_conditional_losses_412879ї
!fruit_tn2/StatefulPartitionedCallStatefulPartitionedCall!dropout1/PartitionedCall:output:0fruit_tn2_413175fruit_tn2_413177fruit_tn2_413179fruit_tn2_413181*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Г*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_fruit_tn2_layer_call_and_return_conditional_losses_413174–
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
GPU2*0J 8В *:
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
valueB:з
+fruit_tn2/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn2/ActivityRegularizer/Shape:output:0:fruit_tn2/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
"fruit_tn2/ActivityRegularizer/CastCast4fruit_tn2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ±
%fruit_tn2/ActivityRegularizer/truedivRealDiv6fruit_tn2/ActivityRegularizer/PartitionedCall:output:0&fruit_tn2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: z
IdentityIdentity*fruit_tn2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Гi

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
: щ
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
?:€€€€€€€€€22: : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2F
!conv2dmpo/StatefulPartitionedCall!conv2dmpo/StatefulPartitionedCall2J
#conv2dmpo_1/StatefulPartitionedCall#conv2dmpo_1/StatefulPartitionedCall2F
!fruit_tn1/StatefulPartitionedCall!fruit_tn1/StatefulPartitionedCall2F
!fruit_tn2/StatefulPartitionedCall!fruit_tn2/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs
Ј
J
.__inference_max_pooling2d_layer_call_fn_415588

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_412299Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
В
К
I__inference_fruit_tn2_layer_call_and_return_all_conditional_losses_415734

inputs
unknown:$
	unknown_0:Г
	unknown_1:
	unknown_2:	Г
identity

identity_1ИҐStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Г*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_fruit_tn2_layer_call_and_return_conditional_losses_413174®
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
GPU2*0J 8В *:
f5R3
1__inference_fruit_tn2_activity_regularizer_412353p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ГX

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
:€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Т	
С
,__inference_conv2dmpo_1_layer_call_fn_415608

inputs!
unknown:#
	unknown_0:#
	unknown_1:#
	unknown_2:
	unknown_3:	А
identityИҐStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2dmpo_1_layer_call_and_return_conditional_losses_412543x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
АU
ѕ
G__inference_conv2dmpo_1_layer_call_and_return_conditional_losses_415884

inputs1
readvariableop_resource:3
readvariableop_1_resource:3
readvariableop_2_resource:3
readvariableop_3_resource:0
!reshape_2_readvariableop_resource:	А
identityИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3ҐReshape_2/ReadVariableOpn
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
valueB"             О
Tensordot/transpose	TransposeReadVariableOp_1:value:0!Tensordot/transpose/perm:output:0*
T0*&
_output_shapes
:h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@      А
Tensordot/ReshapeReshapeTensordot/transpose:y:0 Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@s
Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             Р
Tensordot/transpose_1	TransposeReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*&
_output_shapes
:j
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ж
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
value$B""                  Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*.
_output_shapes
:s
Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Т
Tensordot_1/transpose	TransposeReadVariableOp_3:value:0#Tensordot_1/transpose/perm:output:0*
T0*&
_output_shapes
:j
Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ж
Tensordot_1/ReshapeReshapeTensordot_1/transpose:y:0"Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:u
Tensordot_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ц
Tensordot_1/transpose_1	TransposeReadVariableOp_2:value:0%Tensordot_1/transpose_1/perm:output:0*
T0*&
_output_shapes
:l
Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   М
Tensordot_1/Reshape_1ReshapeTensordot_1/transpose_1:y:0$Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:@Г
Tensordot_1/MatMulMatMulTensordot_1/Reshape:output:0Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes

:@r
Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  Й
Tensordot_1ReshapeTensordot_1/MatMul:product:0Tensordot_1/shape:output:0*
T0*.
_output_shapes
:{
Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   Ф
Tensordot_2/transpose	TransposeTensordot:output:0#Tensordot_2/transpose/perm:output:0*
T0*.
_output_shapes
:j
Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"А      З
Tensordot_2/ReshapeReshapeTensordot_2/transpose:y:0"Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes
:	А}
Tensordot_2/transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   Ъ
Tensordot_2/transpose_1	TransposeTensordot_1:output:0%Tensordot_2/transpose_1/perm:output:0*
T0*.
_output_shapes
:l
Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   А   Н
Tensordot_2/Reshape_1ReshapeTensordot_2/transpose_1:y:0$Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	АЕ
Tensordot_2/MatMulMatMulTensordot_2/Reshape:output:0Tensordot_2/Reshape_1:output:0*
T0* 
_output_shapes
:
ААВ
Tensordot_2/shapeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                              Щ
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
"(                         	      О
	transpose	TransposeTensordot_2:output:0transpose/perm:output:0*
T0*>
_output_shapes,
*:(Б
transpose_1/permConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                	               Л
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
valueB:Ќ
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
valueB:„
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
€€€€€€€€€К
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
value(B&"                      В
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
valueB:„
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
valueB:ў
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
€€€€€€€€€Р
concat_1ConcatV2strided_slice_3:output:0concat_1/values_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:j
	Reshape_1Reshapetranspose_2:y:0concat_1:output:0*
T0*'
_output_shapes
:АП
Conv2DConv2DinputsReshape_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
w
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes	
:А*
dtype0`
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      z
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0*
_output_shapes
:	Аl
addAddV2Conv2D:output:0Reshape_2:output:0*
T0*0
_output_shapes
:€€€€€€€€€АP
ReluReluadd:z:0*
T0*0
_output_shapes
:€€€€€€€€€Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€АЂ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^Reshape_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_324
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
К
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
:€€€€€€€€€G
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
 *ђ≈'7I
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
ие
…
E__inference_fruit_tn1_layer_call_and_return_conditional_losses_412856

inputs7
!loop_body_readvariableop_resource:>
#loop_body_readvariableop_1_resource:А9
#loop_body_readvariableop_2_resource:;
%loop_body_add_readvariableop_resource:
identityИҐloop_body/ReadVariableOpҐloop_body/ReadVariableOp_1Ґloop_body/ReadVariableOp_2Ґloop_body/add/ReadVariableOp;
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
valueB:—
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
value	B : Э
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
valueB:Г
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
value	B : †
loop_body/SelectV2SelectV2loop_body/Greater:z:0)loop_body/PlaceholderWithDefault:output:0loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: Y
loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ґ
loop_body/GatherV2GatherV2inputsloop_body/SelectV2:output:0 loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:А~
loop_body/ReadVariableOpReadVariableOp!loop_body_readvariableop_resource*"
_output_shapes
:*
dtype0З
loop_body/ReadVariableOp_1ReadVariableOp#loop_body_readvariableop_1_resource*'
_output_shapes
:А*
dtype0В
loop_body/ReadVariableOp_2ReadVariableOp#loop_body_readvariableop_2_resource*"
_output_shapes
:*
dtype0w
"loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ґ
loop_body/Tensordot/transpose	Transposeloop_body/GatherV2:output:0+loop_body/Tensordot/transpose/perm:output:0*
T0*#
_output_shapes
:Аr
!loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Я
loop_body/Tensordot/ReshapeReshape!loop_body/Tensordot/transpose:y:0*loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	Аy
$loop_body/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ™
loop_body/Tensordot/transpose_1	Transpose loop_body/ReadVariableOp:value:0-loop_body/Tensordot/transpose_1/perm:output:0*
T0*"
_output_shapes
:t
#loop_body/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      §
loop_body/Tensordot/Reshape_1Reshape#loop_body/Tensordot/transpose_1:y:0,loop_body/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:Ь
loop_body/Tensordot/MatMulMatMul$loop_body/Tensordot/Reshape:output:0&loop_body/Tensordot/Reshape_1:output:0*
T0*
_output_shapes
:	Аr
loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            Ъ
loop_body/TensordotReshape$loop_body/Tensordot/MatMul:product:0"loop_body/Tensordot/shape:output:0*
T0*'
_output_shapes
:Аy
$loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ђ
loop_body/Tensordot_1/transpose	Transpose"loop_body/ReadVariableOp_2:value:0-loop_body/Tensordot_1/transpose/perm:output:0*
T0*"
_output_shapes
:t
#loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      §
loop_body/Tensordot_1/ReshapeReshape#loop_body/Tensordot_1/transpose:y:0,loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:v
%loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ґ
loop_body/Tensordot_1/Reshape_1Reshapeloop_body/Tensordot:output:0.loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А(Ґ
loop_body/Tensordot_1/MatMulMatMul&loop_body/Tensordot_1/Reshape:output:0(loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	А(x
loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"               §
loop_body/Tensordot_1Reshape&loop_body/Tensordot_1/MatMul:product:0$loop_body/Tensordot_1/shape:output:0*
T0*+
_output_shapes
:Аt
#loop_body/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      §
loop_body/Tensordot_2/ReshapeReshape"loop_body/ReadVariableOp_1:value:0,loop_body/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes
:	А2Б
$loop_body/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ±
loop_body/Tensordot_2/transpose	Transposeloop_body/Tensordot_1:output:0-loop_body/Tensordot_2/transpose/perm:output:0*
T0*+
_output_shapes
:Аv
%loop_body/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ©
loop_body/Tensordot_2/Reshape_1Reshape#loop_body/Tensordot_2/transpose:y:0.loop_body/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А2°
loop_body/Tensordot_2/MatMulMatMul&loop_body/Tensordot_2/Reshape:output:0(loop_body/Tensordot_2/Reshape_1:output:0*
T0*
_output_shapes

:p
loop_body/Tensordot_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         Ы
loop_body/Tensordot_2Reshape&loop_body/Tensordot_2/MatMul:product:0$loop_body/Tensordot_2/shape:output:0*
T0*"
_output_shapes
:m
loop_body/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Р
loop_body/transpose	Transposeloop_body/Tensordot_2:output:0!loop_body/transpose/perm:output:0*
T0*"
_output_shapes
:Ж
loop_body/add/ReadVariableOpReadVariableOp%loop_body_add_readvariableop_resource*"
_output_shapes
:*
dtype0В
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
:€€€€€€€€€^
loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : _
loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ф
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
value	B :Ъ
loop_body/SelectV2/pfor/add_1AddV2'loop_body/SelectV2/pfor/Rank_2:output:0(loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: Х
loop_body/SelectV2/pfor/MaximumMaximum'loop_body/SelectV2/pfor/Rank_1:output:0loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: Х
!loop_body/SelectV2/pfor/Maximum_1Maximum!loop_body/SelectV2/pfor/add_1:z:0#loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: `
loop_body/SelectV2/pfor/ShapeShapepfor/range:output:0*
T0*
_output_shapes
:У
loop_body/SelectV2/pfor/subSub%loop_body/SelectV2/pfor/Maximum_1:z:0'loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: o
%loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:†
loop_body/SelectV2/pfor/ReshapeReshapeloop_body/SelectV2/pfor/sub:z:0.loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:l
"loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Ю
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
valueB:«
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
valueB:Ћ
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
value	B : П
loop_body/SelectV2/pfor/concatConcatV2.loop_body/SelectV2/pfor/strided_slice:output:0%loop_body/SelectV2/pfor/Tile:output:00loop_body/SelectV2/pfor/strided_slice_1:output:0,loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ш
!loop_body/SelectV2/pfor/Reshape_1Reshapepfor/range:output:0'loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:€€€€€€€€€Љ
 loop_body/SelectV2/pfor/SelectV2SelectV2loop_body/Greater:z:0*loop_body/SelectV2/pfor/Reshape_1:output:0loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:€€€€€€€€€g
%loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : н
 loop_body/GatherV2/pfor/GatherV2GatherV2inputs)loop_body/SelectV2/pfor/SelectV2:output:0.loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*0
_output_shapes
:€€€€€€€€€Аj
(loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :і
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
value	B : Е
)loop_body/Tensordot/transpose/pfor/concatConcatV2;loop_body/Tensordot/transpose/pfor/concat/values_0:output:0*loop_body/Tensordot/transpose/pfor/add:z:07loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:”
,loop_body/Tensordot/transpose/pfor/Transpose	Transpose)loop_body/GatherV2/pfor/GatherV2:output:02loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аn
,loop_body/Tensordot/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : џ
'loop_body/Tensordot/Reshape/pfor/concatConcatV2pfor/Reshape:output:0*loop_body/Tensordot/Reshape/shape:output:05loop_body/Tensordot/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ќ
(loop_body/Tensordot/Reshape/pfor/ReshapeReshape0loop_body/Tensordot/transpose/pfor/Transpose:y:00loop_body/Tensordot/Reshape/pfor/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€АЖ
%loop_body/Tensordot/MatMul/pfor/ShapeShape1loop_body/Tensordot/Reshape/pfor/Reshape:output:0*
T0*
_output_shapes
:q
/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Џ
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
valueB љ
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
valueB Ѕ
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
valueB Ѕ
)loop_body/Tensordot/MatMul/pfor/Reshape_2Reshape.loop_body/Tensordot/MatMul/pfor/split:output:2:loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ±
#loop_body/Tensordot/MatMul/pfor/mulMul0loop_body/Tensordot/MatMul/pfor/Reshape:output:02loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: ¬
/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack'loop_body/Tensordot/MatMul/pfor/mul:z:02loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:№
)loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape1loop_body/Tensordot/Reshape/pfor/Reshape:output:08loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€Њ
&loop_body/Tensordot/MatMul/pfor/MatMulMatMul2loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0&loop_body/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€|
1loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€З
/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePack0loop_body/Tensordot/MatMul/pfor/Reshape:output:02loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0:loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:„
)loop_body/Tensordot/MatMul/pfor/Reshape_4Reshape0loop_body/Tensordot/MatMul/pfor/MatMul:product:08loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€Аf
$loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : √
loop_body/Tensordot/pfor/concatConcatV2pfor/Reshape:output:0"loop_body/Tensordot/shape:output:0-loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:»
 loop_body/Tensordot/pfor/ReshapeReshape2loop_body/Tensordot/MatMul/pfor/Reshape_4:output:0(loop_body/Tensordot/pfor/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аr
0loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : з
+loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2pfor/Reshape:output:0.loop_body/Tensordot_1/Reshape_1/shape:output:09loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ѕ
,loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape)loop_body/Tensordot/pfor/Reshape:output:04loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А(М
'loop_body/Tensordot_1/MatMul/pfor/ShapeShape5loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
5loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0>loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЋ
)loop_body/Tensordot_1/MatMul/pfor/MinimumMinimum8loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: k
)loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :і
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
valueB"          о
(loop_body/Tensordot_1/MatMul/pfor/SelectSelect+loop_body/Tensordot_1/MatMul/pfor/Equal:z:03loop_body/Tensordot_1/MatMul/pfor/Select/t:output:03loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:Б
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskС
'loop_body/Tensordot_1/MatMul/pfor/stackPack:loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:к
+loop_body/Tensordot_1/MatMul/pfor/transpose	Transpose5loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:01loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€ќ
)loop_body/Tensordot_1/MatMul/pfor/ReshapeReshape/loop_body/Tensordot_1/MatMul/pfor/transpose:y:00loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:€€€€€€€€€А(Л
)loop_body/Tensordot_1/MatMul/pfor/Shape_1Shape2loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:s
1loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : в
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
valueB «
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
valueB «
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
valueB «
+loop_body/Tensordot_1/MatMul/pfor/Reshape_3Reshape0loop_body/Tensordot_1/MatMul/pfor/split:output:2<loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: є
%loop_body/Tensordot_1/MatMul/pfor/mulMul4loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: »
1loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePack4loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:0)loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:б
+loop_body/Tensordot_1/MatMul/pfor/Reshape_4Reshape2loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0:loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€¬
(loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul&loop_body/Tensordot_1/Reshape:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€~
3loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€С
1loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePack<loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:о
+loop_body/Tensordot_1/MatMul/pfor/Reshape_5Reshape2loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0:loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€З
2loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          д
-loop_body/Tensordot_1/MatMul/pfor/transpose_1	Transpose4loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0;loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€А(h
&loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : …
!loop_body/Tensordot_1/pfor/concatConcatV2pfor/Reshape:output:0$loop_body/Tensordot_1/shape:output:0/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ѕ
"loop_body/Tensordot_1/pfor/ReshapeReshape1loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:0*loop_body/Tensordot_1/pfor/concat:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€Аl
*loop_body/Tensordot_2/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ї
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
value	B : Н
+loop_body/Tensordot_2/transpose/pfor/concatConcatV2=loop_body/Tensordot_2/transpose/pfor/concat/values_0:output:0,loop_body/Tensordot_2/transpose/pfor/add:z:09loop_body/Tensordot_2/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:б
.loop_body/Tensordot_2/transpose/pfor/Transpose	Transpose+loop_body/Tensordot_1/pfor/Reshape:output:04loop_body/Tensordot_2/transpose/pfor/concat:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€Аr
0loop_body/Tensordot_2/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : з
+loop_body/Tensordot_2/Reshape_1/pfor/concatConcatV2pfor/Reshape:output:0.loop_body/Tensordot_2/Reshape_1/shape:output:09loop_body/Tensordot_2/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ў
,loop_body/Tensordot_2/Reshape_1/pfor/ReshapeReshape2loop_body/Tensordot_2/transpose/pfor/Transpose:y:04loop_body/Tensordot_2/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2М
'loop_body/Tensordot_2/MatMul/pfor/ShapeShape5loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
5loop_body/Tensordot_2/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
/loop_body/Tensordot_2/MatMul/pfor/strided_sliceStridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0>loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_1StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЋ
)loop_body/Tensordot_2/MatMul/pfor/MinimumMinimum8loop_body/Tensordot_2/MatMul/pfor/strided_slice:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: k
)loop_body/Tensordot_2/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :і
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
valueB"          о
(loop_body/Tensordot_2/MatMul/pfor/SelectSelect+loop_body/Tensordot_2/MatMul/pfor/Equal:z:03loop_body/Tensordot_2/MatMul/pfor/Select/t:output:03loop_body/Tensordot_2/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:Б
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_2StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_3StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_4StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskС
'loop_body/Tensordot_2/MatMul/pfor/stackPack:loop_body/Tensordot_2/MatMul/pfor/strided_slice_2:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_3:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:к
+loop_body/Tensordot_2/MatMul/pfor/transpose	Transpose5loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:01loop_body/Tensordot_2/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€ќ
)loop_body/Tensordot_2/MatMul/pfor/ReshapeReshape/loop_body/Tensordot_2/MatMul/pfor/transpose:y:00loop_body/Tensordot_2/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:А2€€€€€€€€€Л
)loop_body/Tensordot_2/MatMul/pfor/Shape_1Shape2loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:s
1loop_body/Tensordot_2/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : в
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
valueB «
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
valueB «
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
valueB «
+loop_body/Tensordot_2/MatMul/pfor/Reshape_3Reshape0loop_body/Tensordot_2/MatMul/pfor/split:output:2<loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: є
%loop_body/Tensordot_2/MatMul/pfor/mulMul4loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: »
1loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shapePack4loop_body/Tensordot_2/MatMul/pfor/Reshape_1:output:0)loop_body/Tensordot_2/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:б
+loop_body/Tensordot_2/MatMul/pfor/Reshape_4Reshape2loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0:loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€¬
(loop_body/Tensordot_2/MatMul/pfor/MatMulMatMul&loop_body/Tensordot_2/Reshape:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€~
3loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€С
1loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shapePack<loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:о
+loop_body/Tensordot_2/MatMul/pfor/Reshape_5Reshape2loop_body/Tensordot_2/MatMul/pfor/MatMul:product:0:loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€З
2loop_body/Tensordot_2/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          г
-loop_body/Tensordot_2/MatMul/pfor/transpose_1	Transpose4loop_body/Tensordot_2/MatMul/pfor/Reshape_5:output:0;loop_body/Tensordot_2/MatMul/pfor/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€h
&loop_body/Tensordot_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : …
!loop_body/Tensordot_2/pfor/concatConcatV2pfor/Reshape:output:0$loop_body/Tensordot_2/shape:output:0/loop_body/Tensordot_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:∆
"loop_body/Tensordot_2/pfor/ReshapeReshape1loop_body/Tensordot_2/MatMul/pfor/transpose_1:y:0*loop_body/Tensordot_2/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€`
loop_body/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ц
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
value	B : Ё
loop_body/transpose/pfor/concatConcatV21loop_body/transpose/pfor/concat/values_0:output:0 loop_body/transpose/pfor/add:z:0-loop_body/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ј
"loop_body/transpose/pfor/Transpose	Transpose+loop_body/Tensordot_2/pfor/Reshape:output:0(loop_body/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€Y
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
value	B :З
loop_body/add/pfor/addAddV2"loop_body/add/pfor/Rank_1:output:0!loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: Д
loop_body/add/pfor/MaximumMaximumloop_body/add/pfor/add:z:0 loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: n
loop_body/add/pfor/ShapeShape&loop_body/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:А
loop_body/add/pfor/subSubloop_body/add/pfor/Maximum:z:0 loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: j
 loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:С
loop_body/add/pfor/ReshapeReshapeloop_body/add/pfor/sub:z:0)loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:g
loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:П
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
valueB:Ѓ
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
valueB:і
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
value	B : ц
loop_body/add/pfor/concatConcatV2)loop_body/add/pfor/strided_slice:output:0 loop_body/add/pfor/Tile:output:0+loop_body/add/pfor/strided_slice_1:output:0'loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:≠
loop_body/add/pfor/Reshape_1Reshape&loop_body/transpose/pfor/Transpose:y:0"loop_body/add/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€®
loop_body/add/pfor/AddV2AddV2%loop_body/add/pfor/Reshape_1:output:0$loop_body/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   z
ReshapeReshapeloop_body/add/pfor/AddV2:z:0Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@P
ReluReluReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Ї
NoOpNoOp^loop_body/ReadVariableOp^loop_body/ReadVariableOp_1^loop_body/ReadVariableOp_2^loop_body/add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : : : 24
loop_body/ReadVariableOploop_body/ReadVariableOp28
loop_body/ReadVariableOp_1loop_body/ReadVariableOp_128
loop_body/ReadVariableOp_2loop_body/ReadVariableOp_22<
loop_body/add/ReadVariableOploop_body/add/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
С
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_415593

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
К
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
:€€€€€€€€€G
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
 *ђ≈'7I
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
Ђ
џ
*__inference_fruit_tn2_layer_call_fn_415719

inputs
unknown:$
	unknown_0:Г
	unknown_1:
	unknown_2:	Г
identityИҐStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Г*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_fruit_tn2_layer_call_and_return_conditional_losses_413174p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Г`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ї
L
0__inference_max_pooling2d_1_layer_call_fn_415630

inputs
identity№
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_412324Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
М
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
:€€€€€€€€€G
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
 *ђ≈'7I
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
≥9
Љ
E__inference_conv2dmpo_layer_call_and_return_conditional_losses_412438

inputs1
readvariableop_resource:3
readvariableop_1_resource:/
!reshape_2_readvariableop_resource:
identityИҐReadVariableOpҐReadVariableOp_1ҐReshape_2/ReadVariableOpn
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
valueB"             О
Tensordot/transpose	TransposeReadVariableOp_1:value:0!Tensordot/transpose/perm:output:0*
T0*&
_output_shapes
:h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      А
Tensordot/ReshapeReshapeTensordot/transpose:y:0 Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:s
Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             Р
Tensordot/transpose_1	TransposeReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*&
_output_shapes
:j
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ж
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
value$B""                  Г
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
valueB:Ќ
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
valueB:„
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
€€€€€€€€€К
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
valueB:„
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
valueB:ў
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
€€€€€€€€€Р
concat_1ConcatV2strided_slice_3:output:0concat_1/values_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:i
	Reshape_1Reshapetranspose_2:y:0concat_1:output:0*
T0*&
_output_shapes
:О
Conv2DConv2DinputsReshape_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€//*
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
:€€€€€€€€€//O
ReluReluadd:z:0*
T0*/
_output_shapes
:€€€€€€€€€//i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€//Е
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^Reshape_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€//: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_124
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€//
 
_user_specified_nameinputs
£
ƒ
*__inference_conv2dmpo_layer_call_fn_415570

inputs!
unknown:#
	unknown_0:
	unknown_1:
identityИҐStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€//*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2dmpo_layer_call_and_return_conditional_losses_412438w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€//`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€//: : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€//
 
_user_specified_nameinputs
т	
c
D__inference_dropout1_layer_call_and_return_conditional_losses_413290

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¶
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
АU
ѕ
G__inference_conv2dmpo_1_layer_call_and_return_conditional_losses_412543

inputs1
readvariableop_resource:3
readvariableop_1_resource:3
readvariableop_2_resource:3
readvariableop_3_resource:0
!reshape_2_readvariableop_resource:	А
identityИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3ҐReshape_2/ReadVariableOpn
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
valueB"             О
Tensordot/transpose	TransposeReadVariableOp_2:value:0!Tensordot/transpose/perm:output:0*
T0*&
_output_shapes
:h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@      А
Tensordot/ReshapeReshapeTensordot/transpose:y:0 Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@s
Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             Т
Tensordot/transpose_1	TransposeReadVariableOp_3:value:0#Tensordot/transpose_1/perm:output:0*
T0*&
_output_shapes
:j
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ж
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
value$B""                  Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*.
_output_shapes
:s
Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Р
Tensordot_1/transpose	TransposeReadVariableOp:value:0#Tensordot_1/transpose/perm:output:0*
T0*&
_output_shapes
:j
Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ж
Tensordot_1/ReshapeReshapeTensordot_1/transpose:y:0"Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:u
Tensordot_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ц
Tensordot_1/transpose_1	TransposeReadVariableOp_1:value:0%Tensordot_1/transpose_1/perm:output:0*
T0*&
_output_shapes
:l
Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   М
Tensordot_1/Reshape_1ReshapeTensordot_1/transpose_1:y:0$Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:@Г
Tensordot_1/MatMulMatMulTensordot_1/Reshape:output:0Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes

:@r
Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""                  Й
Tensordot_1ReshapeTensordot_1/MatMul:product:0Tensordot_1/shape:output:0*
T0*.
_output_shapes
:{
Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   Ф
Tensordot_2/transpose	TransposeTensordot:output:0#Tensordot_2/transpose/perm:output:0*
T0*.
_output_shapes
:j
Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"А      З
Tensordot_2/ReshapeReshapeTensordot_2/transpose:y:0"Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes
:	А}
Tensordot_2/transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   Ъ
Tensordot_2/transpose_1	TransposeTensordot_1:output:0%Tensordot_2/transpose_1/perm:output:0*
T0*.
_output_shapes
:l
Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   А   Н
Tensordot_2/Reshape_1ReshapeTensordot_2/transpose_1:y:0$Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	АЕ
Tensordot_2/MatMulMatMulTensordot_2/Reshape:output:0Tensordot_2/Reshape_1:output:0*
T0* 
_output_shapes
:
ААВ
Tensordot_2/shapeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                              Щ
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
"(                      	         О
	transpose	TransposeTensordot_2:output:0transpose/perm:output:0*
T0*>
_output_shapes,
*:(Б
transpose_1/permConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                	               Л
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
valueB:Ќ
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
valueB:„
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
€€€€€€€€€К
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
value(B&"                      В
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
valueB:„
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
valueB:ў
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
€€€€€€€€€Р
concat_1ConcatV2strided_slice_3:output:0concat_1/values_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:j
	Reshape_1Reshapetranspose_2:y:0concat_1:output:0*
T0*'
_output_shapes
:АП
Conv2DConv2DinputsReshape_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
w
Reshape_2/ReadVariableOpReadVariableOp!reshape_2_readvariableop_resource*
_output_shapes	
:А*
dtype0`
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      z
	Reshape_2Reshape Reshape_2/ReadVariableOp:value:0Reshape_2/shape:output:0*
T0*
_output_shapes
:	Аl
addAddV2Conv2D:output:0Reshape_2:output:0*
T0*0
_output_shapes
:€€€€€€€€€АP
ReluReluadd:z:0*
T0*0
_output_shapes
:€€€€€€€€€Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€АЂ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^Reshape_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€: : : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_324
Reshape_2/ReadVariableOpReshape_2/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ие
…
E__inference_fruit_tn1_layer_call_and_return_conditional_losses_416176

inputs7
!loop_body_readvariableop_resource:>
#loop_body_readvariableop_1_resource:А9
#loop_body_readvariableop_2_resource:;
%loop_body_add_readvariableop_resource:
identityИҐloop_body/ReadVariableOpҐloop_body/ReadVariableOp_1Ґloop_body/ReadVariableOp_2Ґloop_body/add/ReadVariableOp;
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
valueB:—
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
value	B : Э
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
valueB:Г
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
value	B : †
loop_body/SelectV2SelectV2loop_body/Greater:z:0)loop_body/PlaceholderWithDefault:output:0loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: Y
loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ґ
loop_body/GatherV2GatherV2inputsloop_body/SelectV2:output:0 loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:А~
loop_body/ReadVariableOpReadVariableOp!loop_body_readvariableop_resource*"
_output_shapes
:*
dtype0З
loop_body/ReadVariableOp_1ReadVariableOp#loop_body_readvariableop_1_resource*'
_output_shapes
:А*
dtype0В
loop_body/ReadVariableOp_2ReadVariableOp#loop_body_readvariableop_2_resource*"
_output_shapes
:*
dtype0w
"loop_body/Tensordot/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ґ
loop_body/Tensordot/transpose	Transposeloop_body/GatherV2:output:0+loop_body/Tensordot/transpose/perm:output:0*
T0*#
_output_shapes
:Аr
!loop_body/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Я
loop_body/Tensordot/ReshapeReshape!loop_body/Tensordot/transpose:y:0*loop_body/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	Аy
$loop_body/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ™
loop_body/Tensordot/transpose_1	Transpose loop_body/ReadVariableOp:value:0-loop_body/Tensordot/transpose_1/perm:output:0*
T0*"
_output_shapes
:t
#loop_body/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      §
loop_body/Tensordot/Reshape_1Reshape#loop_body/Tensordot/transpose_1:y:0,loop_body/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:Ь
loop_body/Tensordot/MatMulMatMul$loop_body/Tensordot/Reshape:output:0&loop_body/Tensordot/Reshape_1:output:0*
T0*
_output_shapes
:	Аr
loop_body/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            Ъ
loop_body/TensordotReshape$loop_body/Tensordot/MatMul:product:0"loop_body/Tensordot/shape:output:0*
T0*'
_output_shapes
:Аy
$loop_body/Tensordot_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ђ
loop_body/Tensordot_1/transpose	Transpose"loop_body/ReadVariableOp_2:value:0-loop_body/Tensordot_1/transpose/perm:output:0*
T0*"
_output_shapes
:t
#loop_body/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      §
loop_body/Tensordot_1/ReshapeReshape#loop_body/Tensordot_1/transpose:y:0,loop_body/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:v
%loop_body/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ґ
loop_body/Tensordot_1/Reshape_1Reshapeloop_body/Tensordot:output:0.loop_body/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А(Ґ
loop_body/Tensordot_1/MatMulMatMul&loop_body/Tensordot_1/Reshape:output:0(loop_body/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	А(x
loop_body/Tensordot_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"               §
loop_body/Tensordot_1Reshape&loop_body/Tensordot_1/MatMul:product:0$loop_body/Tensordot_1/shape:output:0*
T0*+
_output_shapes
:Аt
#loop_body/Tensordot_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      §
loop_body/Tensordot_2/ReshapeReshape"loop_body/ReadVariableOp_1:value:0,loop_body/Tensordot_2/Reshape/shape:output:0*
T0*
_output_shapes
:	А2Б
$loop_body/Tensordot_2/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ±
loop_body/Tensordot_2/transpose	Transposeloop_body/Tensordot_1:output:0-loop_body/Tensordot_2/transpose/perm:output:0*
T0*+
_output_shapes
:Аv
%loop_body/Tensordot_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ©
loop_body/Tensordot_2/Reshape_1Reshape#loop_body/Tensordot_2/transpose:y:0.loop_body/Tensordot_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А2°
loop_body/Tensordot_2/MatMulMatMul&loop_body/Tensordot_2/Reshape:output:0(loop_body/Tensordot_2/Reshape_1:output:0*
T0*
_output_shapes

:p
loop_body/Tensordot_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         Ы
loop_body/Tensordot_2Reshape&loop_body/Tensordot_2/MatMul:product:0$loop_body/Tensordot_2/shape:output:0*
T0*"
_output_shapes
:m
loop_body/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Р
loop_body/transpose	Transposeloop_body/Tensordot_2:output:0!loop_body/transpose/perm:output:0*
T0*"
_output_shapes
:Ж
loop_body/add/ReadVariableOpReadVariableOp%loop_body_add_readvariableop_resource*"
_output_shapes
:*
dtype0В
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
:€€€€€€€€€^
loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : _
loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ф
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
value	B :Ъ
loop_body/SelectV2/pfor/add_1AddV2'loop_body/SelectV2/pfor/Rank_2:output:0(loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: Х
loop_body/SelectV2/pfor/MaximumMaximum'loop_body/SelectV2/pfor/Rank_1:output:0loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: Х
!loop_body/SelectV2/pfor/Maximum_1Maximum!loop_body/SelectV2/pfor/add_1:z:0#loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: `
loop_body/SelectV2/pfor/ShapeShapepfor/range:output:0*
T0*
_output_shapes
:У
loop_body/SelectV2/pfor/subSub%loop_body/SelectV2/pfor/Maximum_1:z:0'loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: o
%loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:†
loop_body/SelectV2/pfor/ReshapeReshapeloop_body/SelectV2/pfor/sub:z:0.loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:l
"loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Ю
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
valueB:«
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
valueB:Ћ
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
value	B : П
loop_body/SelectV2/pfor/concatConcatV2.loop_body/SelectV2/pfor/strided_slice:output:0%loop_body/SelectV2/pfor/Tile:output:00loop_body/SelectV2/pfor/strided_slice_1:output:0,loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ш
!loop_body/SelectV2/pfor/Reshape_1Reshapepfor/range:output:0'loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:€€€€€€€€€Љ
 loop_body/SelectV2/pfor/SelectV2SelectV2loop_body/Greater:z:0*loop_body/SelectV2/pfor/Reshape_1:output:0loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:€€€€€€€€€g
%loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : н
 loop_body/GatherV2/pfor/GatherV2GatherV2inputs)loop_body/SelectV2/pfor/SelectV2:output:0.loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*0
_output_shapes
:€€€€€€€€€Аj
(loop_body/Tensordot/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :і
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
value	B : Е
)loop_body/Tensordot/transpose/pfor/concatConcatV2;loop_body/Tensordot/transpose/pfor/concat/values_0:output:0*loop_body/Tensordot/transpose/pfor/add:z:07loop_body/Tensordot/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:”
,loop_body/Tensordot/transpose/pfor/Transpose	Transpose)loop_body/GatherV2/pfor/GatherV2:output:02loop_body/Tensordot/transpose/pfor/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€Аn
,loop_body/Tensordot/Reshape/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : џ
'loop_body/Tensordot/Reshape/pfor/concatConcatV2pfor/Reshape:output:0*loop_body/Tensordot/Reshape/shape:output:05loop_body/Tensordot/Reshape/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ќ
(loop_body/Tensordot/Reshape/pfor/ReshapeReshape0loop_body/Tensordot/transpose/pfor/Transpose:y:00loop_body/Tensordot/Reshape/pfor/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€АЖ
%loop_body/Tensordot/MatMul/pfor/ShapeShape1loop_body/Tensordot/Reshape/pfor/Reshape:output:0*
T0*
_output_shapes
:q
/loop_body/Tensordot/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Џ
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
valueB љ
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
valueB Ѕ
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
valueB Ѕ
)loop_body/Tensordot/MatMul/pfor/Reshape_2Reshape.loop_body/Tensordot/MatMul/pfor/split:output:2:loop_body/Tensordot/MatMul/pfor/Reshape_2/shape_1:output:0*
T0*
_output_shapes
: ±
#loop_body/Tensordot/MatMul/pfor/mulMul0loop_body/Tensordot/MatMul/pfor/Reshape:output:02loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0*
T0*
_output_shapes
: ¬
/loop_body/Tensordot/MatMul/pfor/Reshape_3/shapePack'loop_body/Tensordot/MatMul/pfor/mul:z:02loop_body/Tensordot/MatMul/pfor/Reshape_2:output:0*
N*
T0*
_output_shapes
:№
)loop_body/Tensordot/MatMul/pfor/Reshape_3Reshape1loop_body/Tensordot/Reshape/pfor/Reshape:output:08loop_body/Tensordot/MatMul/pfor/Reshape_3/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€Њ
&loop_body/Tensordot/MatMul/pfor/MatMulMatMul2loop_body/Tensordot/MatMul/pfor/Reshape_3:output:0&loop_body/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€|
1loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€З
/loop_body/Tensordot/MatMul/pfor/Reshape_4/shapePack0loop_body/Tensordot/MatMul/pfor/Reshape:output:02loop_body/Tensordot/MatMul/pfor/Reshape_1:output:0:loop_body/Tensordot/MatMul/pfor/Reshape_4/shape/2:output:0*
N*
T0*
_output_shapes
:„
)loop_body/Tensordot/MatMul/pfor/Reshape_4Reshape0loop_body/Tensordot/MatMul/pfor/MatMul:product:08loop_body/Tensordot/MatMul/pfor/Reshape_4/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€Аf
$loop_body/Tensordot/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : √
loop_body/Tensordot/pfor/concatConcatV2pfor/Reshape:output:0"loop_body/Tensordot/shape:output:0-loop_body/Tensordot/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:»
 loop_body/Tensordot/pfor/ReshapeReshape2loop_body/Tensordot/MatMul/pfor/Reshape_4:output:0(loop_body/Tensordot/pfor/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аr
0loop_body/Tensordot_1/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : з
+loop_body/Tensordot_1/Reshape_1/pfor/concatConcatV2pfor/Reshape:output:0.loop_body/Tensordot_1/Reshape_1/shape:output:09loop_body/Tensordot_1/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ѕ
,loop_body/Tensordot_1/Reshape_1/pfor/ReshapeReshape)loop_body/Tensordot/pfor/Reshape:output:04loop_body/Tensordot_1/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А(М
'loop_body/Tensordot_1/MatMul/pfor/ShapeShape5loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
5loop_body/Tensordot_1/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
/loop_body/Tensordot_1/MatMul/pfor/strided_sliceStridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0>loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_1:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_1StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЋ
)loop_body/Tensordot_1/MatMul/pfor/MinimumMinimum8loop_body/Tensordot_1/MatMul/pfor/strided_slice:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: k
)loop_body/Tensordot_1/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :і
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
valueB"          о
(loop_body/Tensordot_1/MatMul/pfor/SelectSelect+loop_body/Tensordot_1/MatMul/pfor/Equal:z:03loop_body/Tensordot_1/MatMul/pfor/Select/t:output:03loop_body/Tensordot_1/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:Б
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_2StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_3StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_1/MatMul/pfor/strided_slice_4StridedSlice0loop_body/Tensordot_1/MatMul/pfor/Shape:output:0@loop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_1:output:0Bloop_body/Tensordot_1/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskС
'loop_body/Tensordot_1/MatMul/pfor/stackPack:loop_body/Tensordot_1/MatMul/pfor/strided_slice_2:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_3:output:0:loop_body/Tensordot_1/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:к
+loop_body/Tensordot_1/MatMul/pfor/transpose	Transpose5loop_body/Tensordot_1/Reshape_1/pfor/Reshape:output:01loop_body/Tensordot_1/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€ќ
)loop_body/Tensordot_1/MatMul/pfor/ReshapeReshape/loop_body/Tensordot_1/MatMul/pfor/transpose:y:00loop_body/Tensordot_1/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:€€€€€€€€€А(Л
)loop_body/Tensordot_1/MatMul/pfor/Shape_1Shape2loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:s
1loop_body/Tensordot_1/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : в
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
valueB «
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
valueB «
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
valueB «
+loop_body/Tensordot_1/MatMul/pfor/Reshape_3Reshape0loop_body/Tensordot_1/MatMul/pfor/split:output:2<loop_body/Tensordot_1/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: є
%loop_body/Tensordot_1/MatMul/pfor/mulMul4loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: »
1loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shapePack4loop_body/Tensordot_1/MatMul/pfor/Reshape_1:output:0)loop_body/Tensordot_1/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:б
+loop_body/Tensordot_1/MatMul/pfor/Reshape_4Reshape2loop_body/Tensordot_1/MatMul/pfor/Reshape:output:0:loop_body/Tensordot_1/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€¬
(loop_body/Tensordot_1/MatMul/pfor/MatMulMatMul&loop_body/Tensordot_1/Reshape:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€~
3loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€С
1loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shapePack<loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape/0:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_1/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:о
+loop_body/Tensordot_1/MatMul/pfor/Reshape_5Reshape2loop_body/Tensordot_1/MatMul/pfor/MatMul:product:0:loop_body/Tensordot_1/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€З
2loop_body/Tensordot_1/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          д
-loop_body/Tensordot_1/MatMul/pfor/transpose_1	Transpose4loop_body/Tensordot_1/MatMul/pfor/Reshape_5:output:0;loop_body/Tensordot_1/MatMul/pfor/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€А(h
&loop_body/Tensordot_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : …
!loop_body/Tensordot_1/pfor/concatConcatV2pfor/Reshape:output:0$loop_body/Tensordot_1/shape:output:0/loop_body/Tensordot_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ѕ
"loop_body/Tensordot_1/pfor/ReshapeReshape1loop_body/Tensordot_1/MatMul/pfor/transpose_1:y:0*loop_body/Tensordot_1/pfor/concat:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€Аl
*loop_body/Tensordot_2/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ї
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
value	B : Н
+loop_body/Tensordot_2/transpose/pfor/concatConcatV2=loop_body/Tensordot_2/transpose/pfor/concat/values_0:output:0,loop_body/Tensordot_2/transpose/pfor/add:z:09loop_body/Tensordot_2/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:б
.loop_body/Tensordot_2/transpose/pfor/Transpose	Transpose+loop_body/Tensordot_1/pfor/Reshape:output:04loop_body/Tensordot_2/transpose/pfor/concat:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€Аr
0loop_body/Tensordot_2/Reshape_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : з
+loop_body/Tensordot_2/Reshape_1/pfor/concatConcatV2pfor/Reshape:output:0.loop_body/Tensordot_2/Reshape_1/shape:output:09loop_body/Tensordot_2/Reshape_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ў
,loop_body/Tensordot_2/Reshape_1/pfor/ReshapeReshape2loop_body/Tensordot_2/transpose/pfor/Transpose:y:04loop_body/Tensordot_2/Reshape_1/pfor/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2М
'loop_body/Tensordot_2/MatMul/pfor/ShapeShape5loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:0*
T0*
_output_shapes
:
5loop_body/Tensordot_2/MatMul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
/loop_body/Tensordot_2/MatMul/pfor/strided_sliceStridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0>loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_1:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_1StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЋ
)loop_body/Tensordot_2/MatMul/pfor/MinimumMinimum8loop_body/Tensordot_2/MatMul/pfor/strided_slice:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_1:output:0*
T0*
_output_shapes
: k
)loop_body/Tensordot_2/MatMul/pfor/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :і
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
valueB"          о
(loop_body/Tensordot_2/MatMul/pfor/SelectSelect+loop_body/Tensordot_2/MatMul/pfor/Equal:z:03loop_body/Tensordot_2/MatMul/pfor/Select/t:output:03loop_body/Tensordot_2/MatMul/pfor/Select/e:output:0*
T0*
_output_shapes
:Б
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_2StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_3StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
7loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Г
9loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
1loop_body/Tensordot_2/MatMul/pfor/strided_slice_4StridedSlice0loop_body/Tensordot_2/MatMul/pfor/Shape:output:0@loop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_1:output:0Bloop_body/Tensordot_2/MatMul/pfor/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskС
'loop_body/Tensordot_2/MatMul/pfor/stackPack:loop_body/Tensordot_2/MatMul/pfor/strided_slice_2:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_3:output:0:loop_body/Tensordot_2/MatMul/pfor/strided_slice_4:output:0*
N*
T0*
_output_shapes
:к
+loop_body/Tensordot_2/MatMul/pfor/transpose	Transpose5loop_body/Tensordot_2/Reshape_1/pfor/Reshape:output:01loop_body/Tensordot_2/MatMul/pfor/Select:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€ќ
)loop_body/Tensordot_2/MatMul/pfor/ReshapeReshape/loop_body/Tensordot_2/MatMul/pfor/transpose:y:00loop_body/Tensordot_2/MatMul/pfor/stack:output:0*
T0*,
_output_shapes
:А2€€€€€€€€€Л
)loop_body/Tensordot_2/MatMul/pfor/Shape_1Shape2loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0*
T0*
_output_shapes
:s
1loop_body/Tensordot_2/MatMul/pfor/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : в
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
valueB «
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
valueB «
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
valueB «
+loop_body/Tensordot_2/MatMul/pfor/Reshape_3Reshape0loop_body/Tensordot_2/MatMul/pfor/split:output:2<loop_body/Tensordot_2/MatMul/pfor/Reshape_3/shape_1:output:0*
T0*
_output_shapes
: є
%loop_body/Tensordot_2/MatMul/pfor/mulMul4loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
T0*
_output_shapes
: »
1loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shapePack4loop_body/Tensordot_2/MatMul/pfor/Reshape_1:output:0)loop_body/Tensordot_2/MatMul/pfor/mul:z:0*
N*
T0*
_output_shapes
:б
+loop_body/Tensordot_2/MatMul/pfor/Reshape_4Reshape2loop_body/Tensordot_2/MatMul/pfor/Reshape:output:0:loop_body/Tensordot_2/MatMul/pfor/Reshape_4/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€¬
(loop_body/Tensordot_2/MatMul/pfor/MatMulMatMul&loop_body/Tensordot_2/Reshape:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_4:output:0*
T0*'
_output_shapes
:€€€€€€€€€~
3loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€С
1loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shapePack<loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape/0:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_2:output:04loop_body/Tensordot_2/MatMul/pfor/Reshape_3:output:0*
N*
T0*
_output_shapes
:о
+loop_body/Tensordot_2/MatMul/pfor/Reshape_5Reshape2loop_body/Tensordot_2/MatMul/pfor/MatMul:product:0:loop_body/Tensordot_2/MatMul/pfor/Reshape_5/shape:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€З
2loop_body/Tensordot_2/MatMul/pfor/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          г
-loop_body/Tensordot_2/MatMul/pfor/transpose_1	Transpose4loop_body/Tensordot_2/MatMul/pfor/Reshape_5:output:0;loop_body/Tensordot_2/MatMul/pfor/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€h
&loop_body/Tensordot_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : …
!loop_body/Tensordot_2/pfor/concatConcatV2pfor/Reshape:output:0$loop_body/Tensordot_2/shape:output:0/loop_body/Tensordot_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:∆
"loop_body/Tensordot_2/pfor/ReshapeReshape1loop_body/Tensordot_2/MatMul/pfor/transpose_1:y:0*loop_body/Tensordot_2/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€`
loop_body/transpose/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ц
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
value	B : Ё
loop_body/transpose/pfor/concatConcatV21loop_body/transpose/pfor/concat/values_0:output:0 loop_body/transpose/pfor/add:z:0-loop_body/transpose/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ј
"loop_body/transpose/pfor/Transpose	Transpose+loop_body/Tensordot_2/pfor/Reshape:output:0(loop_body/transpose/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€Y
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
value	B :З
loop_body/add/pfor/addAddV2"loop_body/add/pfor/Rank_1:output:0!loop_body/add/pfor/add/y:output:0*
T0*
_output_shapes
: Д
loop_body/add/pfor/MaximumMaximumloop_body/add/pfor/add:z:0 loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: n
loop_body/add/pfor/ShapeShape&loop_body/transpose/pfor/Transpose:y:0*
T0*
_output_shapes
:А
loop_body/add/pfor/subSubloop_body/add/pfor/Maximum:z:0 loop_body/add/pfor/Rank:output:0*
T0*
_output_shapes
: j
 loop_body/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:С
loop_body/add/pfor/ReshapeReshapeloop_body/add/pfor/sub:z:0)loop_body/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:g
loop_body/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:П
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
valueB:Ѓ
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
valueB:і
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
value	B : ц
loop_body/add/pfor/concatConcatV2)loop_body/add/pfor/strided_slice:output:0 loop_body/add/pfor/Tile:output:0+loop_body/add/pfor/strided_slice_1:output:0'loop_body/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:≠
loop_body/add/pfor/Reshape_1Reshape&loop_body/transpose/pfor/Transpose:y:0"loop_body/add/pfor/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€®
loop_body/add/pfor/AddV2AddV2%loop_body/add/pfor/Reshape_1:output:0$loop_body/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   z
ReshapeReshapeloop_body/add/pfor/AddV2:z:0Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@P
ReluReluReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Ї
NoOpNoOp^loop_body/ReadVariableOp^loop_body/ReadVariableOp_1^loop_body/ReadVariableOp_2^loop_body/add/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : : : 24
loop_body/ReadVariableOploop_body/ReadVariableOp28
loop_body/ReadVariableOp_1loop_body/ReadVariableOp_128
loop_body/ReadVariableOp_2loop_body/ReadVariableOp_22<
loop_body/add/ReadVariableOploop_body/add/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Л
њ
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
	unknown_8:	А
	unknown_9:%

unknown_10:А 

unknown_11: 

unknown_12:

unknown_13:%

unknown_14:Г

unknown_15:

unknown_16:	Г
identityИҐStatefulPartitionedCall∆
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
:€€€€€€€€€Г: : : : *4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_413197p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Г`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€22: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:€€€€€€€€€22
"
_user_specified_name
input_12
„
b
D__inference_dropout1_layer_call_and_return_conditional_losses_415686

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
н_
—
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
conv2dmpo_1_413472:	А&
fruit_tn1_413484:+
fruit_tn1_413486:А&
fruit_tn1_413488:&
fruit_tn1_413490:"
fruit_tn2_413502:+
fruit_tn2_413504:Г"
fruit_tn2_413506:
fruit_tn2_413508:	Г
identity

identity_1

identity_2

identity_3

identity_4ИҐconv2d/StatefulPartitionedCallҐ!conv2dmpo/StatefulPartitionedCallҐ#conv2dmpo_1/StatefulPartitionedCallҐ dropout1/StatefulPartitionedCallҐ!fruit_tn1/StatefulPartitionedCallҐ!fruit_tn2/StatefulPartitionedCallу
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_413443conv2d_413445*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€//*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_412370і
!conv2dmpo/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2dmpo_413448conv2dmpo_413450conv2dmpo_413452*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€//*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2dmpo_layer_call_and_return_conditional_losses_412438–
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
GPU2*0J 8В *:
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
valueB:з
+conv2dmpo/ActivityRegularizer/strided_sliceStridedSlice,conv2dmpo/ActivityRegularizer/Shape:output:0:conv2dmpo/ActivityRegularizer/strided_slice/stack:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_1:output:0<conv2dmpo/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
"conv2dmpo/ActivityRegularizer/CastCast4conv2dmpo/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ±
%conv2dmpo/ActivityRegularizer/truedivRealDiv6conv2dmpo/ActivityRegularizer/PartitionedCall:output:0&conv2dmpo/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: с
max_pooling2d/PartitionedCallPartitionedCall*conv2dmpo/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_412299к
#conv2dmpo_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2dmpo_1_413464conv2dmpo_1_413466conv2dmpo_1_413468conv2dmpo_1_413470conv2dmpo_1_413472*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2dmpo_1_layer_call_and_return_conditional_losses_412543÷
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
GPU2*0J 8В *<
f7R5
3__inference_conv2dmpo_1_activity_regularizer_412315Б
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
valueB:с
-conv2dmpo_1/ActivityRegularizer/strided_sliceStridedSlice.conv2dmpo_1/ActivityRegularizer/Shape:output:0<conv2dmpo_1/ActivityRegularizer/strided_slice/stack:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_1:output:0>conv2dmpo_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskФ
$conv2dmpo_1/ActivityRegularizer/CastCast6conv2dmpo_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ј
'conv2dmpo_1/ActivityRegularizer/truedivRealDiv8conv2dmpo_1/ActivityRegularizer/PartitionedCall:output:0(conv2dmpo_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ш
max_pooling2d_1/PartitionedCallPartitionedCall,conv2dmpo_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_412324Ѕ
!fruit_tn1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0fruit_tn1_413484fruit_tn1_413486fruit_tn1_413488fruit_tn1_413490*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_fruit_tn1_layer_call_and_return_conditional_losses_412856–
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
GPU2*0J 8В *:
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
valueB:з
+fruit_tn1/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn1/ActivityRegularizer/Shape:output:0:fruit_tn1/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
"fruit_tn1/ActivityRegularizer/CastCast4fruit_tn1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ±
%fruit_tn1/ActivityRegularizer/truedivRealDiv6fruit_tn1/ActivityRegularizer/PartitionedCall:output:0&fruit_tn1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: п
 dropout1/StatefulPartitionedCallStatefulPartitionedCall*fruit_tn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout1_layer_call_and_return_conditional_losses_413290√
!fruit_tn2/StatefulPartitionedCallStatefulPartitionedCall)dropout1/StatefulPartitionedCall:output:0fruit_tn2_413502fruit_tn2_413504fruit_tn2_413506fruit_tn2_413508*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Г*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_fruit_tn2_layer_call_and_return_conditional_losses_413174–
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
GPU2*0J 8В *:
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
valueB:з
+fruit_tn2/ActivityRegularizer/strided_sliceStridedSlice,fruit_tn2/ActivityRegularizer/Shape:output:0:fruit_tn2/ActivityRegularizer/strided_slice/stack:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_1:output:0<fruit_tn2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
"fruit_tn2/ActivityRegularizer/CastCast4fruit_tn2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ±
%fruit_tn2/ActivityRegularizer/truedivRealDiv6fruit_tn2/ActivityRegularizer/PartitionedCall:output:0&fruit_tn2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: z
IdentityIdentity*fruit_tn2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Гi

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
: Ь
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
?:€€€€€€€€€22: : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2F
!conv2dmpo/StatefulPartitionedCall!conv2dmpo/StatefulPartitionedCall2J
#conv2dmpo_1/StatefulPartitionedCall#conv2dmpo_1/StatefulPartitionedCall2D
 dropout1/StatefulPartitionedCall dropout1/StatefulPartitionedCall2F
!fruit_tn1/StatefulPartitionedCall!fruit_tn1/StatefulPartitionedCall2F
!fruit_tn2/StatefulPartitionedCall!fruit_tn2/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs
¶

ы
B__inference_conv2d_layer_call_and_return_conditional_losses_412370

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€//*
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
:€€€€€€€€€//g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€//w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs
т	
c
D__inference_dropout1_layer_call_and_return_conditional_losses_415698

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¶
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
 
к
*__inference_fruit_tn1_layer_call_fn_415656

inputs
unknown:$
	unknown_0:А
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_fruit_tn1_layer_call_and_return_conditional_losses_412856o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
У
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_415635

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Л
њ
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
	unknown_8:	А
	unknown_9:%

unknown_10:А 

unknown_11: 

unknown_12:

unknown_13:%

unknown_14:Г

unknown_15:

unknown_16:	Г
identityИҐStatefulPartitionedCall∆
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
:€€€€€€€€€Г: : : : *4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_413524p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Г`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€22: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:€€€€€€€€€22
"
_user_specified_name
input_12"џL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ј
serving_default£
E
input_129
serving_default_input_12:0€€€€€€€€€22>
	fruit_tn21
StatefulPartitionedCall:0€€€€€€€€€Гtensorflow/serving/predict:—…
–
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
ї

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
п
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
•
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
Х
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
•
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
–
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
Љ
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J_random_generator
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
–
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
Т
Witer

Xbeta_1

Ybeta_2
	ZdecaymЬmЭmЮmЯm†+m°,mҐ-m£.m§/m•<m¶=mІ>m®?m©Mm™NmЂOmђPm≠vЃvѓv∞v±v≤+v≥,vі-vµ.vґ/vЈ<vЄ=vє>vЇ?vїMvЉNvљOvЊPvњvhatјvhatЅvhat¬vhat√vhatƒ+vhat≈,vhat∆-vhat«.vhat»/vhat…<vhat =vhatЋ>vhatћ?vhatЌMvhatќNvhatѕOvhat–Pvhat—"
	optimizer
¶
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
¶
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
 
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
В2€
-__inference_sequential_1_layer_call_fn_413240
-__inference_sequential_1_layer_call_fn_413861
-__inference_sequential_1_layer_call_fn_413906
-__inference_sequential_1_layer_call_fn_413612ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
о2л
H__inference_sequential_1_layer_call_and_return_conditional_losses_414698
H__inference_sequential_1_layer_call_and_return_conditional_losses_415497
H__inference_sequential_1_layer_call_and_return_conditional_losses_413696
H__inference_sequential_1_layer_call_and_return_conditional_losses_413780ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЌB 
!__inference__wrapped_model_412277input_12"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
≠
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
—2ќ
'__inference_conv2d_layer_call_fn_415549Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_conv2d_layer_call_and_return_conditional_losses_415559Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
 
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
‘2—
*__inference_conv2dmpo_layer_call_fn_415570Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
у2р
I__inference_conv2dmpo_layer_call_and_return_all_conditional_losses_415583Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
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
Ў2’
.__inference_max_pooling2d_layer_call_fn_415588Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
у2р
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_415593Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
:А2conv2dmpo_1/bias
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
 
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
÷2”
,__inference_conv2dmpo_1_layer_call_fn_415608Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
х2т
K__inference_conv2dmpo_1_layer_call_and_return_all_conditional_losses_415625Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
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
Џ2„
0__inference_max_pooling2d_1_layer_call_fn_415630Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
х2т
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_415635Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
:2a
:А2b
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
ѕ
~non_trainable_variables

layers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
Гactivity_regularizer_fn
*E&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
‘2—
*__inference_fruit_tn1_layer_call_fn_415656Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
у2р
I__inference_fruit_tn1_layer_call_and_return_all_conditional_losses_415671Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Р2Н
)__inference_dropout1_layer_call_fn_415676
)__inference_dropout1_layer_call_fn_415681і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
∆2√
D__inference_dropout1_layer_call_and_return_conditional_losses_415686
D__inference_dropout1_layer_call_and_return_conditional_losses_415698і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
:2a
:Г2b
:2c
:Г2bias
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
—
Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
Пactivity_regularizer_fn
*V&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
‘2—
*__inference_fruit_tn2_layer_call_fn_415719Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
у2р
I__inference_fruit_tn2_layer_call_and_return_all_conditional_losses_415734Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
С0
Т1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ћB…
$__inference_signature_wrapper_415540input_12"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
в2я
1__inference_conv2dmpo_activity_regularizer_412290©
Ф≤Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ
	К
п2м
E__inference_conv2dmpo_layer_call_and_return_conditional_losses_415796Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
д2б
3__inference_conv2dmpo_1_activity_regularizer_412315©
Ф≤Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ
	К
с2о
G__inference_conv2dmpo_1_layer_call_and_return_conditional_losses_415884Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
в2я
1__inference_fruit_tn1_activity_regularizer_412340©
Ф≤Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ
	К
п2м
E__inference_fruit_tn1_layer_call_and_return_conditional_losses_416176Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
в2я
1__inference_fruit_tn2_activity_regularizer_412353©
Ф≤Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ
	К
п2м
E__inference_fruit_tn2_layer_call_and_return_conditional_losses_416469Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
R

Уtotal

Фcount
Х	variables
Ц	keras_api"
_tf_keras_metric
v
Ч
thresholds
Шtrue_positives
Щfalse_positives
Ъ	variables
Ы	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
У0
Ф1"
trackable_list_wrapper
.
Х	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
Ш0
Щ1"
trackable_list_wrapper
.
Ъ	variables"
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
$:"А2Adam/conv2dmpo_1/bias/m
:2Adam/a/m
!:А2Adam/b/m
:2Adam/c/m
:2Adam/bias/m
:2Adam/a/m
!:Г2Adam/b/m
:2Adam/c/m
:Г2Adam/bias/m
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
7:52Adam/conv2dmpo/end_node_first/v
6:42Adam/conv2dmpo/end_node_last/v
!:2Adam/conv2dmpo/bias/v
9:72!Adam/conv2dmpo_1/end_node_first/v
8:62 Adam/conv2dmpo_1/middle_node_0/v
8:62 Adam/conv2dmpo_1/middle_node_1/v
8:62 Adam/conv2dmpo_1/end_node_last/v
$:"А2Adam/conv2dmpo_1/bias/v
:2Adam/a/v
!:А2Adam/b/v
:2Adam/c/v
:2Adam/bias/v
:2Adam/a/v
!:Г2Adam/b/v
:2Adam/c/v
:Г2Adam/bias/v
/:-2Adam/conv2d/kernel/vhat
!:2Adam/conv2d/bias/vhat
::82"Adam/conv2dmpo/end_node_first/vhat
9:72!Adam/conv2dmpo/end_node_last/vhat
$:"2Adam/conv2dmpo/bias/vhat
<::2$Adam/conv2dmpo_1/end_node_first/vhat
;:92#Adam/conv2dmpo_1/middle_node_0/vhat
;:92#Adam/conv2dmpo_1/middle_node_1/vhat
;:92#Adam/conv2dmpo_1/end_node_last/vhat
':%А2Adam/conv2dmpo_1/bias/vhat
:2Adam/a/vhat
$:"А2Adam/b/vhat
:2Adam/c/vhat
": 2Adam/bias/vhat
:2Adam/a/vhat
$:"Г2Adam/b/vhat
:2Adam/c/vhat
:Г2Adam/bias/vhat≠
!__inference__wrapped_model_412277З+,-./<=>?MNOP9Ґ6
/Ґ,
*К'
input_12€€€€€€€€€22
™ "6™3
1
	fruit_tn2$К!
	fruit_tn2€€€€€€€€€Г≤
B__inference_conv2d_layer_call_and_return_conditional_losses_415559l7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€22
™ "-Ґ*
#К 
0€€€€€€€€€//
Ъ К
'__inference_conv2d_layer_call_fn_415549_7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€22
™ " К€€€€€€€€€//]
3__inference_conv2dmpo_1_activity_regularizer_412315&Ґ
Ґ
К	
x
™ "К Ќ
K__inference_conv2dmpo_1_layer_call_and_return_all_conditional_losses_415625~+,-./7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "<Ґ9
$К!
0€€€€€€€€€А
Ъ
К	
1/0 ї
G__inference_conv2dmpo_1_layer_call_and_return_conditional_losses_415884p+,-./7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ У
,__inference_conv2dmpo_1_layer_call_fn_415608c+,-./7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "!К€€€€€€€€€А[
1__inference_conv2dmpo_activity_regularizer_412290&Ґ
Ґ
К	
x
™ "К »
I__inference_conv2dmpo_layer_call_and_return_all_conditional_losses_415583{7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€//
™ ";Ґ8
#К 
0€€€€€€€€€//
Ъ
К	
1/0 ґ
E__inference_conv2dmpo_layer_call_and_return_conditional_losses_415796m7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€//
™ "-Ґ*
#К 
0€€€€€€€€€//
Ъ О
*__inference_conv2dmpo_layer_call_fn_415570`7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€//
™ " К€€€€€€€€€//§
D__inference_dropout1_layer_call_and_return_conditional_losses_415686\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ §
D__inference_dropout1_layer_call_and_return_conditional_losses_415698\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "%Ґ"
К
0€€€€€€€€€@
Ъ |
)__inference_dropout1_layer_call_fn_415676O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "К€€€€€€€€€@|
)__inference_dropout1_layer_call_fn_415681O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "К€€€€€€€€€@[
1__inference_fruit_tn1_activity_regularizer_412340&Ґ
Ґ
К	
x
™ "К ¬
I__inference_fruit_tn1_layer_call_and_return_all_conditional_losses_415671u<=>?8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "3Ґ0
К
0€€€€€€€€€@
Ъ
К	
1/0 ∞
E__inference_fruit_tn1_layer_call_and_return_conditional_losses_416176g<=>?8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€@
Ъ И
*__inference_fruit_tn1_layer_call_fn_415656Z<=>?8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "К€€€€€€€€€@[
1__inference_fruit_tn2_activity_regularizer_412353&Ґ
Ґ
К	
x
™ "К Ї
I__inference_fruit_tn2_layer_call_and_return_all_conditional_losses_415734mMNOP/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "4Ґ1
К
0€€€€€€€€€Г
Ъ
К	
1/0 ®
E__inference_fruit_tn2_layer_call_and_return_conditional_losses_416469_MNOP/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "&Ґ#
К
0€€€€€€€€€Г
Ъ А
*__inference_fruit_tn2_layer_call_fn_415719RMNOP/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€Го
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_415635ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ∆
0__inference_max_pooling2d_1_layer_call_fn_415630СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€м
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_415593ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ƒ
.__inference_max_pooling2d_layer_call_fn_415588СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Д
H__inference_sequential_1_layer_call_and_return_conditional_losses_413696Ј+,-./<=>?MNOPAҐ>
7Ґ4
*К'
input_12€€€€€€€€€22
p 

 
™ "^Ґ[
К
0€€€€€€€€€Г
;Ъ8
К	
1/0 
К	
1/1 
К	
1/2 
К	
1/3 Д
H__inference_sequential_1_layer_call_and_return_conditional_losses_413780Ј+,-./<=>?MNOPAҐ>
7Ґ4
*К'
input_12€€€€€€€€€22
p

 
™ "^Ґ[
К
0€€€€€€€€€Г
;Ъ8
К	
1/0 
К	
1/1 
К	
1/2 
К	
1/3 В
H__inference_sequential_1_layer_call_and_return_conditional_losses_414698µ+,-./<=>?MNOP?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€22
p 

 
™ "^Ґ[
К
0€€€€€€€€€Г
;Ъ8
К	
1/0 
К	
1/1 
К	
1/2 
К	
1/3 В
H__inference_sequential_1_layer_call_and_return_conditional_losses_415497µ+,-./<=>?MNOP?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€22
p

 
™ "^Ґ[
К
0€€€€€€€€€Г
;Ъ8
К	
1/0 
К	
1/1 
К	
1/2 
К	
1/3 £
-__inference_sequential_1_layer_call_fn_413240r+,-./<=>?MNOPAҐ>
7Ґ4
*К'
input_12€€€€€€€€€22
p 

 
™ "К€€€€€€€€€Г£
-__inference_sequential_1_layer_call_fn_413612r+,-./<=>?MNOPAҐ>
7Ґ4
*К'
input_12€€€€€€€€€22
p

 
™ "К€€€€€€€€€Г°
-__inference_sequential_1_layer_call_fn_413861p+,-./<=>?MNOP?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€22
p 

 
™ "К€€€€€€€€€Г°
-__inference_sequential_1_layer_call_fn_413906p+,-./<=>?MNOP?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€22
p

 
™ "К€€€€€€€€€ГЉ
$__inference_signature_wrapper_415540У+,-./<=>?MNOPEҐB
Ґ 
;™8
6
input_12*К'
input_12€€€€€€€€€22"6™3
1
	fruit_tn2$К!
	fruit_tn2€€€€€€€€€Г