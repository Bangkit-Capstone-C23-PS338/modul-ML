��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
p
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2	"
adj_xbool( "
adj_ybool( 
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
7
Square
x"T
y"T"
Ttype:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
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
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
�
Adam/v/dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/dense_23/bias
y
(Adam/v/dense_23/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_23/bias*
_output_shapes
:@*
dtype0
�
Adam/m/dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/dense_23/bias
y
(Adam/m/dense_23/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_23/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*'
shared_nameAdam/v/dense_23/kernel
�
*Adam/v/dense_23/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_23/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/m/dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*'
shared_nameAdam/m/dense_23/kernel
�
*Adam/m/dense_23/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_23/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/v/dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/v/dense_22/bias
z
(Adam/v/dense_22/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_22/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/m/dense_22/bias
z
(Adam/m/dense_22/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_22/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/v/dense_22/kernel
�
*Adam/v/dense_22/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_22/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/m/dense_22/kernel
�
*Adam/m/dense_22/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_22/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/v/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/v/dense_21/bias
z
(Adam/v/dense_21/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_21/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/m/dense_21/bias
z
(Adam/m/dense_21/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_21/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/v/dense_21/kernel
�
*Adam/v/dense_21/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_21/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/m/dense_21/kernel
�
*Adam/m/dense_21/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_21/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/v/dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/v/dense_20/bias
z
(Adam/v/dense_20/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_20/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/m/dense_20/bias
z
(Adam/m/dense_20/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_20/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/v/dense_20/kernel
�
*Adam/v/dense_20/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_20/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/m/dense_20/kernel
�
*Adam/m/dense_20/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_20/kernel*
_output_shapes
:	�*
dtype0
�
Adam/v/dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/dense_19/bias
y
(Adam/v/dense_19/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_19/bias*
_output_shapes
:@*
dtype0
�
Adam/m/dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/dense_19/bias
y
(Adam/m/dense_19/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_19/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/v/dense_19/kernel
�
*Adam/v/dense_19/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_19/kernel*
_output_shapes

:@@*
dtype0
�
Adam/m/dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/m/dense_19/kernel
�
*Adam/m/dense_19/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_19/kernel*
_output_shapes

:@@*
dtype0
�
Adam/v/dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/dense_18/bias
y
(Adam/v/dense_18/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_18/bias*
_output_shapes
:@*
dtype0
�
Adam/m/dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/dense_18/bias
y
(Adam/m/dense_18/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_18/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*'
shared_nameAdam/v/dense_18/kernel
�
*Adam/v/dense_18/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_18/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/m/dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*'
shared_nameAdam/m/dense_18/kernel
�
*Adam/m/dense_18/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_18/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/v/dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/v/dense_17/bias
z
(Adam/v/dense_17/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_17/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/m/dense_17/bias
z
(Adam/m/dense_17/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_17/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/v/dense_17/kernel
�
*Adam/v/dense_17/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_17/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/m/dense_17/kernel
�
*Adam/m/dense_17/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_17/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/v/dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/v/dense_16/bias
z
(Adam/v/dense_16/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_16/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/m/dense_16/bias
z
(Adam/m/dense_16/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_16/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/v/dense_16/kernel
�
*Adam/v/dense_16/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_16/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/m/dense_16/kernel
�
*Adam/m/dense_16/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_16/kernel*
_output_shapes
:	�*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
:@*
dtype0
{
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@* 
shared_namedense_23/kernel
t
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes
:	�@*
dtype0
s
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_22/bias
l
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes	
:�*
dtype0
|
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_22/kernel
u
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel* 
_output_shapes
:
��*
dtype0
s
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_21/bias
l
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes	
:�*
dtype0
|
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_21/kernel
u
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel* 
_output_shapes
:
��*
dtype0
s
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_20/bias
l
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes	
:�*
dtype0
{
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_20/kernel
t
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes
:	�*
dtype0
r
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_19/bias
k
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes
:@*
dtype0
z
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_19/kernel
s
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes

:@@*
dtype0
r
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_18/bias
k
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes
:@*
dtype0
{
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@* 
shared_namedense_18/kernel
t
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes
:	�@*
dtype0
s
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_17/bias
l
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes	
:�*
dtype0
|
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_17/kernel
u
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel* 
_output_shapes
:
��*
dtype0
s
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_16/bias
l
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes	
:�*
dtype0
{
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_16/kernel
t
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes
:	�*
dtype0
~
serving_default_inf_featurePlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
~
serving_default_own_featurePlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_inf_featureserving_default_own_featuredense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_1203898

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
* 
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
 layer-3
!layer_with_weights-2
!layer-4
"layer_with_weights-3
"layer-5
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses*

)	keras_api* 

*	keras_api* 
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses* 
z
10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13
?14
@15*
z
10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13
?14
@15*
* 
�
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ftrace_0
Gtrace_1
Htrace_2
Itrace_3* 
6
Jtrace_0
Ktrace_1
Ltrace_2
Mtrace_3* 
* 
�
N
_variables
O_iterations
P_learning_rate
Q_index_dict
R
_momentums
S_velocities
T_update_step_xla*

Userving_default* 
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

1kernel
2bias*
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses
b_random_generator* 
�
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

3kernel
4bias*
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses
o_random_generator* 
�
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses

5kernel
6bias*
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses

7kernel
8bias*
<
10
21
32
43
54
65
76
87*
<
10
21
32
43
54
65
76
87*
* 
�
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

9kernel
:bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

;kernel
<bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

=kernel
>bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

?kernel
@bias*
<
90
:1
;2
<3
=4
>5
?6
@7*
<
90
:1
;2
<3
=4
>5
?6
@7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
OI
VARIABLE_VALUEdense_16/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_16/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_17/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_17/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_18/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_18/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_19/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_19/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_20/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_20/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_21/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_21/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_22/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_22/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_23/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_23/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
0
1
2
3
4
5
6*

�0
�1
�2*
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
�
O0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15*
* 
* 

10
21*

10
21*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

30
41*

30
41*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

50
61*

50
61*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

70
81*

70
81*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
.
0
1
2
3
4
5*
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

90
:1*

90
:1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

;0
<1*

;0
<1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

=0
>1*

=0
>1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

?0
@1*

?0
@1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
.
0
1
2
 3
!4
"5*
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
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
a[
VARIABLE_VALUEAdam/m/dense_16/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_16/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_16/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_16/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_17/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_17/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_17/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_17/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_18/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_18/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_18/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_18/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_19/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_19/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_19/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_19/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_20/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_20/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_20/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_20/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_21/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_21/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_21/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_21/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_22/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_22/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_22/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_22/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_23/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_23/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_23/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_23/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
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

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_16/kerneldense_16/biasdense_17/kerneldense_17/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/bias	iterationlearning_rateAdam/m/dense_16/kernelAdam/v/dense_16/kernelAdam/m/dense_16/biasAdam/v/dense_16/biasAdam/m/dense_17/kernelAdam/v/dense_17/kernelAdam/m/dense_17/biasAdam/v/dense_17/biasAdam/m/dense_18/kernelAdam/v/dense_18/kernelAdam/m/dense_18/biasAdam/v/dense_18/biasAdam/m/dense_19/kernelAdam/v/dense_19/kernelAdam/m/dense_19/biasAdam/v/dense_19/biasAdam/m/dense_20/kernelAdam/v/dense_20/kernelAdam/m/dense_20/biasAdam/v/dense_20/biasAdam/m/dense_21/kernelAdam/v/dense_21/kernelAdam/m/dense_21/biasAdam/v/dense_21/biasAdam/m/dense_22/kernelAdam/v/dense_22/kernelAdam/m/dense_22/biasAdam/v/dense_22/biasAdam/m/dense_23/kernelAdam/v/dense_23/kernelAdam/m/dense_23/biasAdam/v/dense_23/biastotal_2count_2total_1count_1totalcountConst*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_1205058
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_16/kerneldense_16/biasdense_17/kerneldense_17/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/bias	iterationlearning_rateAdam/m/dense_16/kernelAdam/v/dense_16/kernelAdam/m/dense_16/biasAdam/v/dense_16/biasAdam/m/dense_17/kernelAdam/v/dense_17/kernelAdam/m/dense_17/biasAdam/v/dense_17/biasAdam/m/dense_18/kernelAdam/v/dense_18/kernelAdam/m/dense_18/biasAdam/v/dense_18/biasAdam/m/dense_19/kernelAdam/v/dense_19/kernelAdam/m/dense_19/biasAdam/v/dense_19/biasAdam/m/dense_20/kernelAdam/v/dense_20/kernelAdam/m/dense_20/biasAdam/v/dense_20/biasAdam/m/dense_21/kernelAdam/v/dense_21/kernelAdam/m/dense_21/biasAdam/v/dense_21/biasAdam/m/dense_22/kernelAdam/v/dense_22/kernelAdam/m/dense_22/biasAdam/v/dense_22/biasAdam/m/dense_23/kernelAdam/v/dense_23/kernelAdam/m/dense_23/biasAdam/v/dense_23/biastotal_2count_2total_1count_1totalcount*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_1205236��
�	
�
E__inference_dense_19_layer_call_and_return_conditional_losses_1202889

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
.__inference_sequential_4_layer_call_fn_1204191

inputs
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_1202961o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�"
#__inference__traced_restore_1205236
file_prefix3
 assignvariableop_dense_16_kernel:	�/
 assignvariableop_1_dense_16_bias:	�6
"assignvariableop_2_dense_17_kernel:
��/
 assignvariableop_3_dense_17_bias:	�5
"assignvariableop_4_dense_18_kernel:	�@.
 assignvariableop_5_dense_18_bias:@4
"assignvariableop_6_dense_19_kernel:@@.
 assignvariableop_7_dense_19_bias:@5
"assignvariableop_8_dense_20_kernel:	�/
 assignvariableop_9_dense_20_bias:	�7
#assignvariableop_10_dense_21_kernel:
��0
!assignvariableop_11_dense_21_bias:	�7
#assignvariableop_12_dense_22_kernel:
��0
!assignvariableop_13_dense_22_bias:	�6
#assignvariableop_14_dense_23_kernel:	�@/
!assignvariableop_15_dense_23_bias:@'
assignvariableop_16_iteration:	 +
!assignvariableop_17_learning_rate: =
*assignvariableop_18_adam_m_dense_16_kernel:	�=
*assignvariableop_19_adam_v_dense_16_kernel:	�7
(assignvariableop_20_adam_m_dense_16_bias:	�7
(assignvariableop_21_adam_v_dense_16_bias:	�>
*assignvariableop_22_adam_m_dense_17_kernel:
��>
*assignvariableop_23_adam_v_dense_17_kernel:
��7
(assignvariableop_24_adam_m_dense_17_bias:	�7
(assignvariableop_25_adam_v_dense_17_bias:	�=
*assignvariableop_26_adam_m_dense_18_kernel:	�@=
*assignvariableop_27_adam_v_dense_18_kernel:	�@6
(assignvariableop_28_adam_m_dense_18_bias:@6
(assignvariableop_29_adam_v_dense_18_bias:@<
*assignvariableop_30_adam_m_dense_19_kernel:@@<
*assignvariableop_31_adam_v_dense_19_kernel:@@6
(assignvariableop_32_adam_m_dense_19_bias:@6
(assignvariableop_33_adam_v_dense_19_bias:@=
*assignvariableop_34_adam_m_dense_20_kernel:	�=
*assignvariableop_35_adam_v_dense_20_kernel:	�7
(assignvariableop_36_adam_m_dense_20_bias:	�7
(assignvariableop_37_adam_v_dense_20_bias:	�>
*assignvariableop_38_adam_m_dense_21_kernel:
��>
*assignvariableop_39_adam_v_dense_21_kernel:
��7
(assignvariableop_40_adam_m_dense_21_bias:	�7
(assignvariableop_41_adam_v_dense_21_bias:	�>
*assignvariableop_42_adam_m_dense_22_kernel:
��>
*assignvariableop_43_adam_v_dense_22_kernel:
��7
(assignvariableop_44_adam_m_dense_22_bias:	�7
(assignvariableop_45_adam_v_dense_22_bias:	�=
*assignvariableop_46_adam_m_dense_23_kernel:	�@=
*assignvariableop_47_adam_v_dense_23_kernel:	�@6
(assignvariableop_48_adam_m_dense_23_bias:@6
(assignvariableop_49_adam_v_dense_23_bias:@%
assignvariableop_50_total_2: %
assignvariableop_51_count_2: %
assignvariableop_52_total_1: %
assignvariableop_53_count_1: #
assignvariableop_54_total: #
assignvariableop_55_count: 
identity_57��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*�
value�B�9B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*�
value|Bz9B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::*G
dtypes=
;29	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_16_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_16_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_17_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_17_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_18_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_18_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_19_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_19_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_20_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_20_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_21_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_21_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_22_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_22_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_23_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_23_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_iterationIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp!assignvariableop_17_learning_rateIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_m_dense_16_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_v_dense_16_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_m_dense_16_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_v_dense_16_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_m_dense_17_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_v_dense_17_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_m_dense_17_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_v_dense_17_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_m_dense_18_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_v_dense_18_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_m_dense_18_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_v_dense_18_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_m_dense_19_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_v_dense_19_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_m_dense_19_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_v_dense_19_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_m_dense_20_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_v_dense_20_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_m_dense_20_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_v_dense_20_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_m_dense_21_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_v_dense_21_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_m_dense_21_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_v_dense_21_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_m_dense_22_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_v_dense_22_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_m_dense_22_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_v_dense_22_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_m_dense_23_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_v_dense_23_kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_m_dense_23_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_v_dense_23_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpassignvariableop_50_total_2Identity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOpassignvariableop_51_count_2Identity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOpassignvariableop_52_total_1Identity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOpassignvariableop_53_count_1Identity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOpassignvariableop_54_totalIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOpassignvariableop_55_countIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_56Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_57IdentityIdentity_56:output:0^NoOp_1*
T0*
_output_shapes
: �

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_57Identity_57:output:0*�
_input_shapest
r: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
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
AssignVariableOp_3AssignVariableOp_32*
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
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
.__inference_sequential_5_layer_call_fn_1204313

inputs
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_1203296o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
E__inference_dense_23_layer_call_and_return_conditional_losses_1203224

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_21_layer_call_and_return_conditional_losses_1203177

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

e
F__inference_dropout_4_layer_call_and_return_conditional_losses_1203164

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   Ae
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
E__inference_dense_23_layer_call_and_return_conditional_losses_1204698

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_20_layer_call_and_return_conditional_losses_1203146

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
G
+__inference_dropout_5_layer_call_fn_1204642

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_1203254a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_17_layer_call_and_return_conditional_losses_1204499

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_22_layer_call_and_return_conditional_losses_1204679

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_dense_17_layer_call_fn_1204488

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_1202842p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
.__inference_sequential_5_layer_call_fn_1203315
dense_20_input
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_1203296o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:���������
(
_user_specified_namedense_20_input
�

�
E__inference_dense_17_layer_call_and_return_conditional_losses_1202842

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
I__inference_sequential_4_layer_call_and_return_conditional_losses_1202896
dense_16_input#
dense_16_1202812:	�
dense_16_1202814:	�$
dense_17_1202843:
��
dense_17_1202845:	�#
dense_18_1202874:	�@
dense_18_1202876:@"
dense_19_1202890:@@
dense_19_1202892:@
identity�� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall�!dropout_2/StatefulPartitionedCall�!dropout_3/StatefulPartitionedCall�
 dense_16/StatefulPartitionedCallStatefulPartitionedCalldense_16_inputdense_16_1202812dense_16_1202814*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_1202811�
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_1202829�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_17_1202843dense_17_1202845*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_1202842�
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_1202860�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_18_1202874dense_18_1202876*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_18_layer_call_and_return_conditional_losses_1202873�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_1202890dense_19_1202892*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_19_layer_call_and_return_conditional_losses_1202889x
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:W S
'
_output_shapes
:���������
(
_user_specified_namedense_16_input
�	
�
.__inference_sequential_5_layer_call_fn_1204334

inputs
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_1203343o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
d
+__inference_dropout_3_layer_call_fn_1204504

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_1202860p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_21_layer_call_and_return_conditional_losses_1204632

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

e
F__inference_dropout_5_layer_call_and_return_conditional_losses_1203195

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   Ae
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
G
+__inference_dropout_3_layer_call_fn_1204509

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_1202919a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
+__inference_dropout_2_layer_call_fn_1204457

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_1202829p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�%
�
I__inference_sequential_4_layer_call_and_return_conditional_losses_1204292

inputs:
'dense_16_matmul_readvariableop_resource:	�7
(dense_16_biasadd_readvariableop_resource:	�;
'dense_17_matmul_readvariableop_resource:
��7
(dense_17_biasadd_readvariableop_resource:	�:
'dense_18_matmul_readvariableop_resource:	�@6
(dense_18_biasadd_readvariableop_resource:@9
'dense_19_matmul_readvariableop_resource:@@6
(dense_19_biasadd_readvariableop_resource:@
identity��dense_16/BiasAdd/ReadVariableOp�dense_16/MatMul/ReadVariableOp�dense_17/BiasAdd/ReadVariableOp�dense_17/MatMul/ReadVariableOp�dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0|
dense_16/MatMulMatMulinputs&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*(
_output_shapes
:����������n
dropout_2/IdentityIdentitydense_16/Relu:activations:0*
T0*(
_output_shapes
:�����������
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_17/MatMulMatMuldropout_2/Identity:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_17/ReluReludense_17/BiasAdd:output:0*
T0*(
_output_shapes
:����������n
dropout_3/IdentityIdentitydense_17/Relu:activations:0*
T0*(
_output_shapes
:�����������
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_18/MatMulMatMuldropout_3/Identity:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@h
IdentityIdentitydense_19/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
.__inference_sequential_4_layer_call_fn_1202980
dense_16_input
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_1202961o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:���������
(
_user_specified_namedense_16_input
�

e
F__inference_dropout_3_layer_call_and_return_conditional_losses_1204521

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   Ae
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
I__inference_sequential_5_layer_call_and_return_conditional_losses_1203231
dense_20_input#
dense_20_1203147:	�
dense_20_1203149:	�$
dense_21_1203178:
��
dense_21_1203180:	�$
dense_22_1203209:
��
dense_22_1203211:	�#
dense_23_1203225:	�@
dense_23_1203227:@
identity�� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall�!dropout_4/StatefulPartitionedCall�!dropout_5/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCalldense_20_inputdense_20_1203147dense_20_1203149*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1203146�
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_4_layer_call_and_return_conditional_losses_1203164�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_21_1203178dense_21_1203180*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1203177�
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_1203195�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_22_1203209dense_22_1203211*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_1203208�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_1203225dense_23_1203227*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_1203224x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:W S
'
_output_shapes
:���������
(
_user_specified_namedense_20_input
�
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_1202908

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_dense_20_layer_call_fn_1204574

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1203146p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
d
+__inference_dropout_4_layer_call_fn_1204590

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_4_layer_call_and_return_conditional_losses_1203164p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_18_layer_call_and_return_conditional_losses_1204546

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�5
�
I__inference_sequential_4_layer_call_and_return_conditional_losses_1204259

inputs:
'dense_16_matmul_readvariableop_resource:	�7
(dense_16_biasadd_readvariableop_resource:	�;
'dense_17_matmul_readvariableop_resource:
��7
(dense_17_biasadd_readvariableop_resource:	�:
'dense_18_matmul_readvariableop_resource:	�@6
(dense_18_biasadd_readvariableop_resource:@9
'dense_19_matmul_readvariableop_resource:@@6
(dense_19_biasadd_readvariableop_resource:@
identity��dense_16/BiasAdd/ReadVariableOp�dense_16/MatMul/ReadVariableOp�dense_17/BiasAdd/ReadVariableOp�dense_17/MatMul/ReadVariableOp�dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0|
dense_16/MatMulMatMulinputs&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*(
_output_shapes
:����������\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A�
dropout_2/dropout/MulMuldense_16/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:����������p
dropout_2/dropout/ShapeShapedense_16/Relu:activations:0*
T0*
_output_shapes
::���
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?�
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������^
dropout_2/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_2/dropout/SelectV2SelectV2"dropout_2/dropout/GreaterEqual:z:0dropout_2/dropout/Mul:z:0"dropout_2/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_17/MatMulMatMul#dropout_2/dropout/SelectV2:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_17/ReluReludense_17/BiasAdd:output:0*
T0*(
_output_shapes
:����������\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A�
dropout_3/dropout/MulMuldense_17/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:����������p
dropout_3/dropout/ShapeShapedense_17/Relu:activations:0*
T0*
_output_shapes
::���
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?�
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������^
dropout_3/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_3/dropout/SelectV2SelectV2"dropout_3/dropout/GreaterEqual:z:0dropout_3/dropout/Mul:z:0"dropout_3/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_18/MatMulMatMul#dropout_3/dropout/SelectV2:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@h
IdentityIdentitydense_19/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_dense_18_layer_call_fn_1204535

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_18_layer_call_and_return_conditional_losses_1202873o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
F__inference_dropout_4_layer_call_and_return_conditional_losses_1204612

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
E__inference_dense_19_layer_call_and_return_conditional_losses_1204565

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�
"__inference__wrapped_model_1202796
inf_feature
own_featureO
<model_2_sequential_5_dense_20_matmul_readvariableop_resource:	�L
=model_2_sequential_5_dense_20_biasadd_readvariableop_resource:	�P
<model_2_sequential_5_dense_21_matmul_readvariableop_resource:
��L
=model_2_sequential_5_dense_21_biasadd_readvariableop_resource:	�P
<model_2_sequential_5_dense_22_matmul_readvariableop_resource:
��L
=model_2_sequential_5_dense_22_biasadd_readvariableop_resource:	�O
<model_2_sequential_5_dense_23_matmul_readvariableop_resource:	�@K
=model_2_sequential_5_dense_23_biasadd_readvariableop_resource:@O
<model_2_sequential_4_dense_16_matmul_readvariableop_resource:	�L
=model_2_sequential_4_dense_16_biasadd_readvariableop_resource:	�P
<model_2_sequential_4_dense_17_matmul_readvariableop_resource:
��L
=model_2_sequential_4_dense_17_biasadd_readvariableop_resource:	�O
<model_2_sequential_4_dense_18_matmul_readvariableop_resource:	�@K
=model_2_sequential_4_dense_18_biasadd_readvariableop_resource:@N
<model_2_sequential_4_dense_19_matmul_readvariableop_resource:@@K
=model_2_sequential_4_dense_19_biasadd_readvariableop_resource:@
identity��4model_2/sequential_4/dense_16/BiasAdd/ReadVariableOp�3model_2/sequential_4/dense_16/MatMul/ReadVariableOp�4model_2/sequential_4/dense_17/BiasAdd/ReadVariableOp�3model_2/sequential_4/dense_17/MatMul/ReadVariableOp�4model_2/sequential_4/dense_18/BiasAdd/ReadVariableOp�3model_2/sequential_4/dense_18/MatMul/ReadVariableOp�4model_2/sequential_4/dense_19/BiasAdd/ReadVariableOp�3model_2/sequential_4/dense_19/MatMul/ReadVariableOp�4model_2/sequential_5/dense_20/BiasAdd/ReadVariableOp�3model_2/sequential_5/dense_20/MatMul/ReadVariableOp�4model_2/sequential_5/dense_21/BiasAdd/ReadVariableOp�3model_2/sequential_5/dense_21/MatMul/ReadVariableOp�4model_2/sequential_5/dense_22/BiasAdd/ReadVariableOp�3model_2/sequential_5/dense_22/MatMul/ReadVariableOp�4model_2/sequential_5/dense_23/BiasAdd/ReadVariableOp�3model_2/sequential_5/dense_23/MatMul/ReadVariableOp�
3model_2/sequential_5/dense_20/MatMul/ReadVariableOpReadVariableOp<model_2_sequential_5_dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
$model_2/sequential_5/dense_20/MatMulMatMulown_feature;model_2/sequential_5/dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
4model_2/sequential_5/dense_20/BiasAdd/ReadVariableOpReadVariableOp=model_2_sequential_5_dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%model_2/sequential_5/dense_20/BiasAddBiasAdd.model_2/sequential_5/dense_20/MatMul:product:0<model_2/sequential_5/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"model_2/sequential_5/dense_20/ReluRelu.model_2/sequential_5/dense_20/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
'model_2/sequential_5/dropout_4/IdentityIdentity0model_2/sequential_5/dense_20/Relu:activations:0*
T0*(
_output_shapes
:�����������
3model_2/sequential_5/dense_21/MatMul/ReadVariableOpReadVariableOp<model_2_sequential_5_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
$model_2/sequential_5/dense_21/MatMulMatMul0model_2/sequential_5/dropout_4/Identity:output:0;model_2/sequential_5/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
4model_2/sequential_5/dense_21/BiasAdd/ReadVariableOpReadVariableOp=model_2_sequential_5_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%model_2/sequential_5/dense_21/BiasAddBiasAdd.model_2/sequential_5/dense_21/MatMul:product:0<model_2/sequential_5/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"model_2/sequential_5/dense_21/ReluRelu.model_2/sequential_5/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
'model_2/sequential_5/dropout_5/IdentityIdentity0model_2/sequential_5/dense_21/Relu:activations:0*
T0*(
_output_shapes
:�����������
3model_2/sequential_5/dense_22/MatMul/ReadVariableOpReadVariableOp<model_2_sequential_5_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
$model_2/sequential_5/dense_22/MatMulMatMul0model_2/sequential_5/dropout_5/Identity:output:0;model_2/sequential_5/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
4model_2/sequential_5/dense_22/BiasAdd/ReadVariableOpReadVariableOp=model_2_sequential_5_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%model_2/sequential_5/dense_22/BiasAddBiasAdd.model_2/sequential_5/dense_22/MatMul:product:0<model_2/sequential_5/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"model_2/sequential_5/dense_22/ReluRelu.model_2/sequential_5/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
3model_2/sequential_5/dense_23/MatMul/ReadVariableOpReadVariableOp<model_2_sequential_5_dense_23_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
$model_2/sequential_5/dense_23/MatMulMatMul0model_2/sequential_5/dense_22/Relu:activations:0;model_2/sequential_5/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
4model_2/sequential_5/dense_23/BiasAdd/ReadVariableOpReadVariableOp=model_2_sequential_5_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
%model_2/sequential_5/dense_23/BiasAddBiasAdd.model_2/sequential_5/dense_23/MatMul:product:0<model_2/sequential_5/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
3model_2/sequential_4/dense_16/MatMul/ReadVariableOpReadVariableOp<model_2_sequential_4_dense_16_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
$model_2/sequential_4/dense_16/MatMulMatMulinf_feature;model_2/sequential_4/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
4model_2/sequential_4/dense_16/BiasAdd/ReadVariableOpReadVariableOp=model_2_sequential_4_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%model_2/sequential_4/dense_16/BiasAddBiasAdd.model_2/sequential_4/dense_16/MatMul:product:0<model_2/sequential_4/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"model_2/sequential_4/dense_16/ReluRelu.model_2/sequential_4/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
'model_2/sequential_4/dropout_2/IdentityIdentity0model_2/sequential_4/dense_16/Relu:activations:0*
T0*(
_output_shapes
:�����������
3model_2/sequential_4/dense_17/MatMul/ReadVariableOpReadVariableOp<model_2_sequential_4_dense_17_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
$model_2/sequential_4/dense_17/MatMulMatMul0model_2/sequential_4/dropout_2/Identity:output:0;model_2/sequential_4/dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
4model_2/sequential_4/dense_17/BiasAdd/ReadVariableOpReadVariableOp=model_2_sequential_4_dense_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%model_2/sequential_4/dense_17/BiasAddBiasAdd.model_2/sequential_4/dense_17/MatMul:product:0<model_2/sequential_4/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"model_2/sequential_4/dense_17/ReluRelu.model_2/sequential_4/dense_17/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
'model_2/sequential_4/dropout_3/IdentityIdentity0model_2/sequential_4/dense_17/Relu:activations:0*
T0*(
_output_shapes
:�����������
3model_2/sequential_4/dense_18/MatMul/ReadVariableOpReadVariableOp<model_2_sequential_4_dense_18_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
$model_2/sequential_4/dense_18/MatMulMatMul0model_2/sequential_4/dropout_3/Identity:output:0;model_2/sequential_4/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
4model_2/sequential_4/dense_18/BiasAdd/ReadVariableOpReadVariableOp=model_2_sequential_4_dense_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
%model_2/sequential_4/dense_18/BiasAddBiasAdd.model_2/sequential_4/dense_18/MatMul:product:0<model_2/sequential_4/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
"model_2/sequential_4/dense_18/ReluRelu.model_2/sequential_4/dense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
3model_2/sequential_4/dense_19/MatMul/ReadVariableOpReadVariableOp<model_2_sequential_4_dense_19_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
$model_2/sequential_4/dense_19/MatMulMatMul0model_2/sequential_4/dense_18/Relu:activations:0;model_2/sequential_4/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
4model_2/sequential_4/dense_19/BiasAdd/ReadVariableOpReadVariableOp=model_2_sequential_4_dense_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
%model_2/sequential_4/dense_19/BiasAddBiasAdd.model_2/sequential_4/dense_19/MatMul:product:0<model_2/sequential_4/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
2model_2/tf.math.l2_normalize_4/l2_normalize/SquareSquare.model_2/sequential_4/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
Amodel_2/tf.math.l2_normalize_4/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
/model_2/tf.math.l2_normalize_4/l2_normalize/SumSum6model_2/tf.math.l2_normalize_4/l2_normalize/Square:y:0Jmodel_2/tf.math.l2_normalize_4/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(z
5model_2/tf.math.l2_normalize_4/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
3model_2/tf.math.l2_normalize_4/l2_normalize/MaximumMaximum8model_2/tf.math.l2_normalize_4/l2_normalize/Sum:output:0>model_2/tf.math.l2_normalize_4/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:����������
1model_2/tf.math.l2_normalize_4/l2_normalize/RsqrtRsqrt7model_2/tf.math.l2_normalize_4/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
+model_2/tf.math.l2_normalize_4/l2_normalizeMul.model_2/sequential_4/dense_19/BiasAdd:output:05model_2/tf.math.l2_normalize_4/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@�
2model_2/tf.math.l2_normalize_5/l2_normalize/SquareSquare.model_2/sequential_5/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
Amodel_2/tf.math.l2_normalize_5/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
/model_2/tf.math.l2_normalize_5/l2_normalize/SumSum6model_2/tf.math.l2_normalize_5/l2_normalize/Square:y:0Jmodel_2/tf.math.l2_normalize_5/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(z
5model_2/tf.math.l2_normalize_5/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
3model_2/tf.math.l2_normalize_5/l2_normalize/MaximumMaximum8model_2/tf.math.l2_normalize_5/l2_normalize/Sum:output:0>model_2/tf.math.l2_normalize_5/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:����������
1model_2/tf.math.l2_normalize_5/l2_normalize/RsqrtRsqrt7model_2/tf.math.l2_normalize_5/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
+model_2/tf.math.l2_normalize_5/l2_normalizeMul.model_2/sequential_5/dense_23/BiasAdd:output:05model_2/tf.math.l2_normalize_5/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@^
model_2/dot_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
model_2/dot_2/ExpandDims
ExpandDims/model_2/tf.math.l2_normalize_4/l2_normalize:z:0%model_2/dot_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������@`
model_2/dot_2/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
model_2/dot_2/ExpandDims_1
ExpandDims/model_2/tf.math.l2_normalize_5/l2_normalize:z:0'model_2/dot_2/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:���������@�
model_2/dot_2/MatMulBatchMatMulV2!model_2/dot_2/ExpandDims:output:0#model_2/dot_2/ExpandDims_1:output:0*
T0*+
_output_shapes
:���������n
model_2/dot_2/ShapeShapemodel_2/dot_2/MatMul:output:0*
T0*
_output_shapes
::���
model_2/dot_2/SqueezeSqueezemodel_2/dot_2/MatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
m
IdentityIdentitymodel_2/dot_2/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp5^model_2/sequential_4/dense_16/BiasAdd/ReadVariableOp4^model_2/sequential_4/dense_16/MatMul/ReadVariableOp5^model_2/sequential_4/dense_17/BiasAdd/ReadVariableOp4^model_2/sequential_4/dense_17/MatMul/ReadVariableOp5^model_2/sequential_4/dense_18/BiasAdd/ReadVariableOp4^model_2/sequential_4/dense_18/MatMul/ReadVariableOp5^model_2/sequential_4/dense_19/BiasAdd/ReadVariableOp4^model_2/sequential_4/dense_19/MatMul/ReadVariableOp5^model_2/sequential_5/dense_20/BiasAdd/ReadVariableOp4^model_2/sequential_5/dense_20/MatMul/ReadVariableOp5^model_2/sequential_5/dense_21/BiasAdd/ReadVariableOp4^model_2/sequential_5/dense_21/MatMul/ReadVariableOp5^model_2/sequential_5/dense_22/BiasAdd/ReadVariableOp4^model_2/sequential_5/dense_22/MatMul/ReadVariableOp5^model_2/sequential_5/dense_23/BiasAdd/ReadVariableOp4^model_2/sequential_5/dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:���������:���������: : : : : : : : : : : : : : : : 2l
4model_2/sequential_4/dense_16/BiasAdd/ReadVariableOp4model_2/sequential_4/dense_16/BiasAdd/ReadVariableOp2j
3model_2/sequential_4/dense_16/MatMul/ReadVariableOp3model_2/sequential_4/dense_16/MatMul/ReadVariableOp2l
4model_2/sequential_4/dense_17/BiasAdd/ReadVariableOp4model_2/sequential_4/dense_17/BiasAdd/ReadVariableOp2j
3model_2/sequential_4/dense_17/MatMul/ReadVariableOp3model_2/sequential_4/dense_17/MatMul/ReadVariableOp2l
4model_2/sequential_4/dense_18/BiasAdd/ReadVariableOp4model_2/sequential_4/dense_18/BiasAdd/ReadVariableOp2j
3model_2/sequential_4/dense_18/MatMul/ReadVariableOp3model_2/sequential_4/dense_18/MatMul/ReadVariableOp2l
4model_2/sequential_4/dense_19/BiasAdd/ReadVariableOp4model_2/sequential_4/dense_19/BiasAdd/ReadVariableOp2j
3model_2/sequential_4/dense_19/MatMul/ReadVariableOp3model_2/sequential_4/dense_19/MatMul/ReadVariableOp2l
4model_2/sequential_5/dense_20/BiasAdd/ReadVariableOp4model_2/sequential_5/dense_20/BiasAdd/ReadVariableOp2j
3model_2/sequential_5/dense_20/MatMul/ReadVariableOp3model_2/sequential_5/dense_20/MatMul/ReadVariableOp2l
4model_2/sequential_5/dense_21/BiasAdd/ReadVariableOp4model_2/sequential_5/dense_21/BiasAdd/ReadVariableOp2j
3model_2/sequential_5/dense_21/MatMul/ReadVariableOp3model_2/sequential_5/dense_21/MatMul/ReadVariableOp2l
4model_2/sequential_5/dense_22/BiasAdd/ReadVariableOp4model_2/sequential_5/dense_22/BiasAdd/ReadVariableOp2j
3model_2/sequential_5/dense_22/MatMul/ReadVariableOp3model_2/sequential_5/dense_22/MatMul/ReadVariableOp2l
4model_2/sequential_5/dense_23/BiasAdd/ReadVariableOp4model_2/sequential_5/dense_23/BiasAdd/ReadVariableOp2j
3model_2/sequential_5/dense_23/MatMul/ReadVariableOp3model_2/sequential_5/dense_23/MatMul/ReadVariableOp:TP
'
_output_shapes
:���������
%
_user_specified_nameown_feature:T P
'
_output_shapes
:���������
%
_user_specified_nameinf_feature
�

�
E__inference_dense_22_layer_call_and_return_conditional_losses_1203208

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_model_2_layer_call_fn_1203681
inf_feature
own_feature
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�@
	unknown_6:@
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�@

unknown_12:@

unknown_13:@@

unknown_14:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinf_featureown_featureunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_1203646o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:���������:���������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:TP
'
_output_shapes
:���������
%
_user_specified_nameown_feature:T P
'
_output_shapes
:���������
%
_user_specified_nameinf_feature
�
�
I__inference_sequential_4_layer_call_and_return_conditional_losses_1203008

inputs#
dense_16_1202985:	�
dense_16_1202987:	�$
dense_17_1202991:
��
dense_17_1202993:	�#
dense_18_1202997:	�@
dense_18_1202999:@"
dense_19_1203002:@@
dense_19_1203004:@
identity�� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall�
 dense_16/StatefulPartitionedCallStatefulPartitionedCallinputsdense_16_1202985dense_16_1202987*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_1202811�
dropout_2/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_1202908�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_17_1202991dense_17_1202993*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_1202842�
dropout_3/PartitionedCallPartitionedCall)dense_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_1202919�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_18_1202997dense_18_1202999*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_18_layer_call_and_return_conditional_losses_1202873�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_1203002dense_19_1203004*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_19_layer_call_and_return_conditional_losses_1202889x
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
l
B__inference_dot_2_layer_call_and_return_conditional_losses_1203531

inputs
inputs_1
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :o

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*+
_output_shapes
:���������@R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :u
ExpandDims_1
ExpandDimsinputs_1ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:���������@y
MatMulBatchMatMulV2ExpandDims:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:���������R
ShapeShapeMatMul:output:0*
T0*
_output_shapes
::��l
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
X
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:OK
'
_output_shapes
:���������@
 
_user_specified_nameinputs:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

e
F__inference_dropout_2_layer_call_and_return_conditional_losses_1202829

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   Ae
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
I__inference_sequential_5_layer_call_and_return_conditional_losses_1203296

inputs#
dense_20_1203273:	�
dense_20_1203275:	�$
dense_21_1203279:
��
dense_21_1203281:	�$
dense_22_1203285:
��
dense_22_1203287:	�#
dense_23_1203290:	�@
dense_23_1203292:@
identity�� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall�!dropout_4/StatefulPartitionedCall�!dropout_5/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCallinputsdense_20_1203273dense_20_1203275*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1203146�
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_4_layer_call_and_return_conditional_losses_1203164�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_21_1203279dense_21_1203281*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1203177�
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_1203195�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_22_1203285dense_22_1203287*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_1203208�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_1203290dense_23_1203292*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_1203224x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�*
�
D__inference_model_2_layer_call_and_return_conditional_losses_1203534
inf_feature
own_feature'
sequential_5_1203471:	�#
sequential_5_1203473:	�(
sequential_5_1203475:
��#
sequential_5_1203477:	�(
sequential_5_1203479:
��#
sequential_5_1203481:	�'
sequential_5_1203483:	�@"
sequential_5_1203485:@'
sequential_4_1203488:	�#
sequential_4_1203490:	�(
sequential_4_1203492:
��#
sequential_4_1203494:	�'
sequential_4_1203496:	�@"
sequential_4_1203498:@&
sequential_4_1203500:@@"
sequential_4_1203502:@
identity��$sequential_4/StatefulPartitionedCall�$sequential_5/StatefulPartitionedCall�
$sequential_5/StatefulPartitionedCallStatefulPartitionedCallown_featuresequential_5_1203471sequential_5_1203473sequential_5_1203475sequential_5_1203477sequential_5_1203479sequential_5_1203481sequential_5_1203483sequential_5_1203485*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_1203296�
$sequential_4/StatefulPartitionedCallStatefulPartitionedCallinf_featuresequential_4_1203488sequential_4_1203490sequential_4_1203492sequential_4_1203494sequential_4_1203496sequential_4_1203498sequential_4_1203500sequential_4_1203502*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_1202961�
*tf.math.l2_normalize_4/l2_normalize/SquareSquare-sequential_4/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������@{
9tf.math.l2_normalize_4/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
'tf.math.l2_normalize_4/l2_normalize/SumSum.tf.math.l2_normalize_4/l2_normalize/Square:y:0Btf.math.l2_normalize_4/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(r
-tf.math.l2_normalize_4/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
+tf.math.l2_normalize_4/l2_normalize/MaximumMaximum0tf.math.l2_normalize_4/l2_normalize/Sum:output:06tf.math.l2_normalize_4/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:����������
)tf.math.l2_normalize_4/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_4/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
#tf.math.l2_normalize_4/l2_normalizeMul-sequential_4/StatefulPartitionedCall:output:0-tf.math.l2_normalize_4/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@�
*tf.math.l2_normalize_5/l2_normalize/SquareSquare-sequential_5/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������@{
9tf.math.l2_normalize_5/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
'tf.math.l2_normalize_5/l2_normalize/SumSum.tf.math.l2_normalize_5/l2_normalize/Square:y:0Btf.math.l2_normalize_5/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(r
-tf.math.l2_normalize_5/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
+tf.math.l2_normalize_5/l2_normalize/MaximumMaximum0tf.math.l2_normalize_5/l2_normalize/Sum:output:06tf.math.l2_normalize_5/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:����������
)tf.math.l2_normalize_5/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_5/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
#tf.math.l2_normalize_5/l2_normalizeMul-sequential_5/StatefulPartitionedCall:output:0-tf.math.l2_normalize_5/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@�
dot_2/PartitionedCallPartitionedCall'tf.math.l2_normalize_4/l2_normalize:z:0'tf.math.l2_normalize_5/l2_normalize:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dot_2_layer_call_and_return_conditional_losses_1203531m
IdentityIdentitydot_2/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:���������:���������: : : : : : : : : : : : : : : : 2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall:TP
'
_output_shapes
:���������
%
_user_specified_nameown_feature:T P
'
_output_shapes
:���������
%
_user_specified_nameinf_feature
�s
�
D__inference_model_2_layer_call_and_return_conditional_losses_1204170
inputs_0
inputs_1G
4sequential_5_dense_20_matmul_readvariableop_resource:	�D
5sequential_5_dense_20_biasadd_readvariableop_resource:	�H
4sequential_5_dense_21_matmul_readvariableop_resource:
��D
5sequential_5_dense_21_biasadd_readvariableop_resource:	�H
4sequential_5_dense_22_matmul_readvariableop_resource:
��D
5sequential_5_dense_22_biasadd_readvariableop_resource:	�G
4sequential_5_dense_23_matmul_readvariableop_resource:	�@C
5sequential_5_dense_23_biasadd_readvariableop_resource:@G
4sequential_4_dense_16_matmul_readvariableop_resource:	�D
5sequential_4_dense_16_biasadd_readvariableop_resource:	�H
4sequential_4_dense_17_matmul_readvariableop_resource:
��D
5sequential_4_dense_17_biasadd_readvariableop_resource:	�G
4sequential_4_dense_18_matmul_readvariableop_resource:	�@C
5sequential_4_dense_18_biasadd_readvariableop_resource:@F
4sequential_4_dense_19_matmul_readvariableop_resource:@@C
5sequential_4_dense_19_biasadd_readvariableop_resource:@
identity��,sequential_4/dense_16/BiasAdd/ReadVariableOp�+sequential_4/dense_16/MatMul/ReadVariableOp�,sequential_4/dense_17/BiasAdd/ReadVariableOp�+sequential_4/dense_17/MatMul/ReadVariableOp�,sequential_4/dense_18/BiasAdd/ReadVariableOp�+sequential_4/dense_18/MatMul/ReadVariableOp�,sequential_4/dense_19/BiasAdd/ReadVariableOp�+sequential_4/dense_19/MatMul/ReadVariableOp�,sequential_5/dense_20/BiasAdd/ReadVariableOp�+sequential_5/dense_20/MatMul/ReadVariableOp�,sequential_5/dense_21/BiasAdd/ReadVariableOp�+sequential_5/dense_21/MatMul/ReadVariableOp�,sequential_5/dense_22/BiasAdd/ReadVariableOp�+sequential_5/dense_22/MatMul/ReadVariableOp�,sequential_5/dense_23/BiasAdd/ReadVariableOp�+sequential_5/dense_23/MatMul/ReadVariableOp�
+sequential_5/dense_20/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_5/dense_20/MatMulMatMulinputs_13sequential_5/dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,sequential_5/dense_20/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_5/dense_20/BiasAddBiasAdd&sequential_5/dense_20/MatMul:product:04sequential_5/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������}
sequential_5/dense_20/ReluRelu&sequential_5/dense_20/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
sequential_5/dropout_4/IdentityIdentity(sequential_5/dense_20/Relu:activations:0*
T0*(
_output_shapes
:�����������
+sequential_5/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_5/dense_21/MatMulMatMul(sequential_5/dropout_4/Identity:output:03sequential_5/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,sequential_5/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_5/dense_21/BiasAddBiasAdd&sequential_5/dense_21/MatMul:product:04sequential_5/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������}
sequential_5/dense_21/ReluRelu&sequential_5/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
sequential_5/dropout_5/IdentityIdentity(sequential_5/dense_21/Relu:activations:0*
T0*(
_output_shapes
:�����������
+sequential_5/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_5/dense_22/MatMulMatMul(sequential_5/dropout_5/Identity:output:03sequential_5/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,sequential_5/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_5/dense_22/BiasAddBiasAdd&sequential_5/dense_22/MatMul:product:04sequential_5/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������}
sequential_5/dense_22/ReluRelu&sequential_5/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
+sequential_5/dense_23/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_23_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
sequential_5/dense_23/MatMulMatMul(sequential_5/dense_22/Relu:activations:03sequential_5/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,sequential_5/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_5/dense_23/BiasAddBiasAdd&sequential_5/dense_23/MatMul:product:04sequential_5/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
+sequential_4/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_16_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_4/dense_16/MatMulMatMulinputs_03sequential_4/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,sequential_4/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_4/dense_16/BiasAddBiasAdd&sequential_4/dense_16/MatMul:product:04sequential_4/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������}
sequential_4/dense_16/ReluRelu&sequential_4/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
sequential_4/dropout_2/IdentityIdentity(sequential_4/dense_16/Relu:activations:0*
T0*(
_output_shapes
:�����������
+sequential_4/dense_17/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_17_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_4/dense_17/MatMulMatMul(sequential_4/dropout_2/Identity:output:03sequential_4/dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,sequential_4/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_4/dense_17/BiasAddBiasAdd&sequential_4/dense_17/MatMul:product:04sequential_4/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������}
sequential_4/dense_17/ReluRelu&sequential_4/dense_17/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
sequential_4/dropout_3/IdentityIdentity(sequential_4/dense_17/Relu:activations:0*
T0*(
_output_shapes
:�����������
+sequential_4/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_18_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
sequential_4/dense_18/MatMulMatMul(sequential_4/dropout_3/Identity:output:03sequential_4/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,sequential_4/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_4/dense_18/BiasAddBiasAdd&sequential_4/dense_18/MatMul:product:04sequential_4/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
sequential_4/dense_18/ReluRelu&sequential_4/dense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
+sequential_4/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_19_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
sequential_4/dense_19/MatMulMatMul(sequential_4/dense_18/Relu:activations:03sequential_4/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,sequential_4/dense_19/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_4/dense_19/BiasAddBiasAdd&sequential_4/dense_19/MatMul:product:04sequential_4/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*tf.math.l2_normalize_4/l2_normalize/SquareSquare&sequential_4/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:���������@{
9tf.math.l2_normalize_4/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
'tf.math.l2_normalize_4/l2_normalize/SumSum.tf.math.l2_normalize_4/l2_normalize/Square:y:0Btf.math.l2_normalize_4/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(r
-tf.math.l2_normalize_4/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
+tf.math.l2_normalize_4/l2_normalize/MaximumMaximum0tf.math.l2_normalize_4/l2_normalize/Sum:output:06tf.math.l2_normalize_4/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:����������
)tf.math.l2_normalize_4/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_4/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
#tf.math.l2_normalize_4/l2_normalizeMul&sequential_4/dense_19/BiasAdd:output:0-tf.math.l2_normalize_4/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@�
*tf.math.l2_normalize_5/l2_normalize/SquareSquare&sequential_5/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:���������@{
9tf.math.l2_normalize_5/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
'tf.math.l2_normalize_5/l2_normalize/SumSum.tf.math.l2_normalize_5/l2_normalize/Square:y:0Btf.math.l2_normalize_5/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(r
-tf.math.l2_normalize_5/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
+tf.math.l2_normalize_5/l2_normalize/MaximumMaximum0tf.math.l2_normalize_5/l2_normalize/Sum:output:06tf.math.l2_normalize_5/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:����������
)tf.math.l2_normalize_5/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_5/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
#tf.math.l2_normalize_5/l2_normalizeMul&sequential_5/dense_23/BiasAdd:output:0-tf.math.l2_normalize_5/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@V
dot_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot_2/ExpandDims
ExpandDims'tf.math.l2_normalize_4/l2_normalize:z:0dot_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������@X
dot_2/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot_2/ExpandDims_1
ExpandDims'tf.math.l2_normalize_5/l2_normalize:z:0dot_2/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:���������@�
dot_2/MatMulBatchMatMulV2dot_2/ExpandDims:output:0dot_2/ExpandDims_1:output:0*
T0*+
_output_shapes
:���������^
dot_2/ShapeShapedot_2/MatMul:output:0*
T0*
_output_shapes
::��x
dot_2/SqueezeSqueezedot_2/MatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
e
IdentityIdentitydot_2/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp-^sequential_4/dense_16/BiasAdd/ReadVariableOp,^sequential_4/dense_16/MatMul/ReadVariableOp-^sequential_4/dense_17/BiasAdd/ReadVariableOp,^sequential_4/dense_17/MatMul/ReadVariableOp-^sequential_4/dense_18/BiasAdd/ReadVariableOp,^sequential_4/dense_18/MatMul/ReadVariableOp-^sequential_4/dense_19/BiasAdd/ReadVariableOp,^sequential_4/dense_19/MatMul/ReadVariableOp-^sequential_5/dense_20/BiasAdd/ReadVariableOp,^sequential_5/dense_20/MatMul/ReadVariableOp-^sequential_5/dense_21/BiasAdd/ReadVariableOp,^sequential_5/dense_21/MatMul/ReadVariableOp-^sequential_5/dense_22/BiasAdd/ReadVariableOp,^sequential_5/dense_22/MatMul/ReadVariableOp-^sequential_5/dense_23/BiasAdd/ReadVariableOp,^sequential_5/dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:���������:���������: : : : : : : : : : : : : : : : 2\
,sequential_4/dense_16/BiasAdd/ReadVariableOp,sequential_4/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_16/MatMul/ReadVariableOp+sequential_4/dense_16/MatMul/ReadVariableOp2\
,sequential_4/dense_17/BiasAdd/ReadVariableOp,sequential_4/dense_17/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_17/MatMul/ReadVariableOp+sequential_4/dense_17/MatMul/ReadVariableOp2\
,sequential_4/dense_18/BiasAdd/ReadVariableOp,sequential_4/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_18/MatMul/ReadVariableOp+sequential_4/dense_18/MatMul/ReadVariableOp2\
,sequential_4/dense_19/BiasAdd/ReadVariableOp,sequential_4/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_19/MatMul/ReadVariableOp+sequential_4/dense_19/MatMul/ReadVariableOp2\
,sequential_5/dense_20/BiasAdd/ReadVariableOp,sequential_5/dense_20/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_20/MatMul/ReadVariableOp+sequential_5/dense_20/MatMul/ReadVariableOp2\
,sequential_5/dense_21/BiasAdd/ReadVariableOp,sequential_5/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_21/MatMul/ReadVariableOp+sequential_5/dense_21/MatMul/ReadVariableOp2\
,sequential_5/dense_22/BiasAdd/ReadVariableOp,sequential_5/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_22/MatMul/ReadVariableOp+sequential_5/dense_22/MatMul/ReadVariableOp2\
,sequential_5/dense_23/BiasAdd/ReadVariableOp,sequential_5/dense_23/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_23/MatMul/ReadVariableOp+sequential_5/dense_23/MatMul/ReadVariableOp:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0
�
d
+__inference_dropout_5_layer_call_fn_1204637

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_1203195p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
G
+__inference_dropout_4_layer_call_fn_1204595

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_4_layer_call_and_return_conditional_losses_1203243a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�5
�
I__inference_sequential_5_layer_call_and_return_conditional_losses_1204381

inputs:
'dense_20_matmul_readvariableop_resource:	�7
(dense_20_biasadd_readvariableop_resource:	�;
'dense_21_matmul_readvariableop_resource:
��7
(dense_21_biasadd_readvariableop_resource:	�;
'dense_22_matmul_readvariableop_resource:
��7
(dense_22_biasadd_readvariableop_resource:	�:
'dense_23_matmul_readvariableop_resource:	�@6
(dense_23_biasadd_readvariableop_resource:@
identity��dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0|
dense_20/MatMulMatMulinputs&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*(
_output_shapes
:����������\
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A�
dropout_4/dropout/MulMuldense_20/Relu:activations:0 dropout_4/dropout/Const:output:0*
T0*(
_output_shapes
:����������p
dropout_4/dropout/ShapeShapedense_20/Relu:activations:0*
T0*
_output_shapes
::���
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?�
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������^
dropout_4/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_4/dropout/SelectV2SelectV2"dropout_4/dropout/GreaterEqual:z:0dropout_4/dropout/Mul:z:0"dropout_4/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_21/MatMulMatMul#dropout_4/dropout/SelectV2:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*(
_output_shapes
:����������\
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A�
dropout_5/dropout/MulMuldense_21/Relu:activations:0 dropout_5/dropout/Const:output:0*
T0*(
_output_shapes
:����������p
dropout_5/dropout/ShapeShapedense_21/Relu:activations:0*
T0*
_output_shapes
::���
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0e
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?�
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������^
dropout_5/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_5/dropout/SelectV2SelectV2"dropout_5/dropout/GreaterEqual:z:0dropout_5/dropout/Mul:z:0"dropout_5/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_22/MatMulMatMul#dropout_5/dropout/SelectV2:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_23/MatMulMatMuldense_22/Relu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@h
IdentityIdentitydense_23/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_sequential_4_layer_call_and_return_conditional_losses_1202961

inputs#
dense_16_1202938:	�
dense_16_1202940:	�$
dense_17_1202944:
��
dense_17_1202946:	�#
dense_18_1202950:	�@
dense_18_1202952:@"
dense_19_1202955:@@
dense_19_1202957:@
identity�� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall�!dropout_2/StatefulPartitionedCall�!dropout_3/StatefulPartitionedCall�
 dense_16/StatefulPartitionedCallStatefulPartitionedCallinputsdense_16_1202938dense_16_1202940*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_1202811�
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_1202829�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_17_1202944dense_17_1202946*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_1202842�
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_1202860�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_18_1202950dense_18_1202952*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_18_layer_call_and_return_conditional_losses_1202873�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_1202955dense_19_1202957*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_19_layer_call_and_return_conditional_losses_1202889x
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

e
F__inference_dropout_4_layer_call_and_return_conditional_losses_1204607

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   Ae
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
n
B__inference_dot_2_layer_call_and_return_conditional_losses_1204432
inputs_0
inputs_1
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :q

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������@R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :u
ExpandDims_1
ExpandDimsinputs_1ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:���������@y
MatMulBatchMatMulV2ExpandDims:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:���������R
ShapeShapeMatMul:output:0*
T0*
_output_shapes
::��l
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
X
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:QM
'
_output_shapes
:���������@
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs_0
�
�
)__inference_model_2_layer_call_fn_1203773
inf_feature
own_feature
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�@
	unknown_6:@
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�@

unknown_12:@

unknown_13:@@

unknown_14:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinf_featureown_featureunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_1203738o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:���������:���������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:TP
'
_output_shapes
:���������
%
_user_specified_nameown_feature:T P
'
_output_shapes
:���������
%
_user_specified_nameinf_feature
�
�
*__inference_dense_23_layer_call_fn_1204688

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_1203224o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
F__inference_dropout_5_layer_call_and_return_conditional_losses_1204659

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
F__inference_dropout_5_layer_call_and_return_conditional_losses_1203254

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
F__inference_dropout_3_layer_call_and_return_conditional_losses_1204526

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
F__inference_dropout_3_layer_call_and_return_conditional_losses_1202919

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

e
F__inference_dropout_5_layer_call_and_return_conditional_losses_1204654

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   Ae
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_dense_21_layer_call_fn_1204621

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1203177p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
I__inference_sequential_5_layer_call_and_return_conditional_losses_1203267
dense_20_input#
dense_20_1203234:	�
dense_20_1203236:	�$
dense_21_1203245:
��
dense_21_1203247:	�$
dense_22_1203256:
��
dense_22_1203258:	�#
dense_23_1203261:	�@
dense_23_1203263:@
identity�� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCalldense_20_inputdense_20_1203234dense_20_1203236*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1203146�
dropout_4/PartitionedCallPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_4_layer_call_and_return_conditional_losses_1203243�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_21_1203245dense_21_1203247*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1203177�
dropout_5/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_1203254�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_22_1203256dense_22_1203258*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_1203208�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_1203261dense_23_1203263*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_1203224x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:W S
'
_output_shapes
:���������
(
_user_specified_namedense_20_input
�	
�
.__inference_sequential_5_layer_call_fn_1203362
dense_20_input
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_1203343o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:���������
(
_user_specified_namedense_20_input
�	
�
.__inference_sequential_4_layer_call_fn_1203027
dense_16_input
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_1203008o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:���������
(
_user_specified_namedense_16_input
�
�
I__inference_sequential_4_layer_call_and_return_conditional_losses_1202932
dense_16_input#
dense_16_1202899:	�
dense_16_1202901:	�$
dense_17_1202910:
��
dense_17_1202912:	�#
dense_18_1202921:	�@
dense_18_1202923:@"
dense_19_1202926:@@
dense_19_1202928:@
identity�� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall�
 dense_16/StatefulPartitionedCallStatefulPartitionedCalldense_16_inputdense_16_1202899dense_16_1202901*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_1202811�
dropout_2/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_1202908�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_17_1202910dense_17_1202912*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_1202842�
dropout_3/PartitionedCallPartitionedCall)dense_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_1202919�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_18_1202921dense_18_1202923*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_18_layer_call_and_return_conditional_losses_1202873�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_1202926dense_19_1202928*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_19_layer_call_and_return_conditional_losses_1202889x
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall:W S
'
_output_shapes
:���������
(
_user_specified_namedense_16_input
��
�
D__inference_model_2_layer_call_and_return_conditional_losses_1204086
inputs_0
inputs_1G
4sequential_5_dense_20_matmul_readvariableop_resource:	�D
5sequential_5_dense_20_biasadd_readvariableop_resource:	�H
4sequential_5_dense_21_matmul_readvariableop_resource:
��D
5sequential_5_dense_21_biasadd_readvariableop_resource:	�H
4sequential_5_dense_22_matmul_readvariableop_resource:
��D
5sequential_5_dense_22_biasadd_readvariableop_resource:	�G
4sequential_5_dense_23_matmul_readvariableop_resource:	�@C
5sequential_5_dense_23_biasadd_readvariableop_resource:@G
4sequential_4_dense_16_matmul_readvariableop_resource:	�D
5sequential_4_dense_16_biasadd_readvariableop_resource:	�H
4sequential_4_dense_17_matmul_readvariableop_resource:
��D
5sequential_4_dense_17_biasadd_readvariableop_resource:	�G
4sequential_4_dense_18_matmul_readvariableop_resource:	�@C
5sequential_4_dense_18_biasadd_readvariableop_resource:@F
4sequential_4_dense_19_matmul_readvariableop_resource:@@C
5sequential_4_dense_19_biasadd_readvariableop_resource:@
identity��,sequential_4/dense_16/BiasAdd/ReadVariableOp�+sequential_4/dense_16/MatMul/ReadVariableOp�,sequential_4/dense_17/BiasAdd/ReadVariableOp�+sequential_4/dense_17/MatMul/ReadVariableOp�,sequential_4/dense_18/BiasAdd/ReadVariableOp�+sequential_4/dense_18/MatMul/ReadVariableOp�,sequential_4/dense_19/BiasAdd/ReadVariableOp�+sequential_4/dense_19/MatMul/ReadVariableOp�,sequential_5/dense_20/BiasAdd/ReadVariableOp�+sequential_5/dense_20/MatMul/ReadVariableOp�,sequential_5/dense_21/BiasAdd/ReadVariableOp�+sequential_5/dense_21/MatMul/ReadVariableOp�,sequential_5/dense_22/BiasAdd/ReadVariableOp�+sequential_5/dense_22/MatMul/ReadVariableOp�,sequential_5/dense_23/BiasAdd/ReadVariableOp�+sequential_5/dense_23/MatMul/ReadVariableOp�
+sequential_5/dense_20/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_5/dense_20/MatMulMatMulinputs_13sequential_5/dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,sequential_5/dense_20/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_5/dense_20/BiasAddBiasAdd&sequential_5/dense_20/MatMul:product:04sequential_5/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������}
sequential_5/dense_20/ReluRelu&sequential_5/dense_20/BiasAdd:output:0*
T0*(
_output_shapes
:����������i
$sequential_5/dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A�
"sequential_5/dropout_4/dropout/MulMul(sequential_5/dense_20/Relu:activations:0-sequential_5/dropout_4/dropout/Const:output:0*
T0*(
_output_shapes
:�����������
$sequential_5/dropout_4/dropout/ShapeShape(sequential_5/dense_20/Relu:activations:0*
T0*
_output_shapes
::���
;sequential_5/dropout_4/dropout/random_uniform/RandomUniformRandomUniform-sequential_5/dropout_4/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0r
-sequential_5/dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?�
+sequential_5/dropout_4/dropout/GreaterEqualGreaterEqualDsequential_5/dropout_4/dropout/random_uniform/RandomUniform:output:06sequential_5/dropout_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������k
&sequential_5/dropout_4/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
'sequential_5/dropout_4/dropout/SelectV2SelectV2/sequential_5/dropout_4/dropout/GreaterEqual:z:0&sequential_5/dropout_4/dropout/Mul:z:0/sequential_5/dropout_4/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
+sequential_5/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_5/dense_21/MatMulMatMul0sequential_5/dropout_4/dropout/SelectV2:output:03sequential_5/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,sequential_5/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_5/dense_21/BiasAddBiasAdd&sequential_5/dense_21/MatMul:product:04sequential_5/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������}
sequential_5/dense_21/ReluRelu&sequential_5/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:����������i
$sequential_5/dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A�
"sequential_5/dropout_5/dropout/MulMul(sequential_5/dense_21/Relu:activations:0-sequential_5/dropout_5/dropout/Const:output:0*
T0*(
_output_shapes
:�����������
$sequential_5/dropout_5/dropout/ShapeShape(sequential_5/dense_21/Relu:activations:0*
T0*
_output_shapes
::���
;sequential_5/dropout_5/dropout/random_uniform/RandomUniformRandomUniform-sequential_5/dropout_5/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0r
-sequential_5/dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?�
+sequential_5/dropout_5/dropout/GreaterEqualGreaterEqualDsequential_5/dropout_5/dropout/random_uniform/RandomUniform:output:06sequential_5/dropout_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������k
&sequential_5/dropout_5/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
'sequential_5/dropout_5/dropout/SelectV2SelectV2/sequential_5/dropout_5/dropout/GreaterEqual:z:0&sequential_5/dropout_5/dropout/Mul:z:0/sequential_5/dropout_5/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
+sequential_5/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_5/dense_22/MatMulMatMul0sequential_5/dropout_5/dropout/SelectV2:output:03sequential_5/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,sequential_5/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_5/dense_22/BiasAddBiasAdd&sequential_5/dense_22/MatMul:product:04sequential_5/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������}
sequential_5/dense_22/ReluRelu&sequential_5/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
+sequential_5/dense_23/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_23_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
sequential_5/dense_23/MatMulMatMul(sequential_5/dense_22/Relu:activations:03sequential_5/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,sequential_5/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_5/dense_23/BiasAddBiasAdd&sequential_5/dense_23/MatMul:product:04sequential_5/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
+sequential_4/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_16_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_4/dense_16/MatMulMatMulinputs_03sequential_4/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,sequential_4/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_4/dense_16/BiasAddBiasAdd&sequential_4/dense_16/MatMul:product:04sequential_4/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������}
sequential_4/dense_16/ReluRelu&sequential_4/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:����������i
$sequential_4/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A�
"sequential_4/dropout_2/dropout/MulMul(sequential_4/dense_16/Relu:activations:0-sequential_4/dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:�����������
$sequential_4/dropout_2/dropout/ShapeShape(sequential_4/dense_16/Relu:activations:0*
T0*
_output_shapes
::���
;sequential_4/dropout_2/dropout/random_uniform/RandomUniformRandomUniform-sequential_4/dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0r
-sequential_4/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?�
+sequential_4/dropout_2/dropout/GreaterEqualGreaterEqualDsequential_4/dropout_2/dropout/random_uniform/RandomUniform:output:06sequential_4/dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������k
&sequential_4/dropout_2/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
'sequential_4/dropout_2/dropout/SelectV2SelectV2/sequential_4/dropout_2/dropout/GreaterEqual:z:0&sequential_4/dropout_2/dropout/Mul:z:0/sequential_4/dropout_2/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
+sequential_4/dense_17/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_17_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_4/dense_17/MatMulMatMul0sequential_4/dropout_2/dropout/SelectV2:output:03sequential_4/dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,sequential_4/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_4/dense_17/BiasAddBiasAdd&sequential_4/dense_17/MatMul:product:04sequential_4/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������}
sequential_4/dense_17/ReluRelu&sequential_4/dense_17/BiasAdd:output:0*
T0*(
_output_shapes
:����������i
$sequential_4/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A�
"sequential_4/dropout_3/dropout/MulMul(sequential_4/dense_17/Relu:activations:0-sequential_4/dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:�����������
$sequential_4/dropout_3/dropout/ShapeShape(sequential_4/dense_17/Relu:activations:0*
T0*
_output_shapes
::���
;sequential_4/dropout_3/dropout/random_uniform/RandomUniformRandomUniform-sequential_4/dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0r
-sequential_4/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?�
+sequential_4/dropout_3/dropout/GreaterEqualGreaterEqualDsequential_4/dropout_3/dropout/random_uniform/RandomUniform:output:06sequential_4/dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������k
&sequential_4/dropout_3/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
'sequential_4/dropout_3/dropout/SelectV2SelectV2/sequential_4/dropout_3/dropout/GreaterEqual:z:0&sequential_4/dropout_3/dropout/Mul:z:0/sequential_4/dropout_3/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
+sequential_4/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_18_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
sequential_4/dense_18/MatMulMatMul0sequential_4/dropout_3/dropout/SelectV2:output:03sequential_4/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,sequential_4/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_4/dense_18/BiasAddBiasAdd&sequential_4/dense_18/MatMul:product:04sequential_4/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
sequential_4/dense_18/ReluRelu&sequential_4/dense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
+sequential_4/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_19_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
sequential_4/dense_19/MatMulMatMul(sequential_4/dense_18/Relu:activations:03sequential_4/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,sequential_4/dense_19/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_4/dense_19/BiasAddBiasAdd&sequential_4/dense_19/MatMul:product:04sequential_4/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*tf.math.l2_normalize_4/l2_normalize/SquareSquare&sequential_4/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:���������@{
9tf.math.l2_normalize_4/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
'tf.math.l2_normalize_4/l2_normalize/SumSum.tf.math.l2_normalize_4/l2_normalize/Square:y:0Btf.math.l2_normalize_4/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(r
-tf.math.l2_normalize_4/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
+tf.math.l2_normalize_4/l2_normalize/MaximumMaximum0tf.math.l2_normalize_4/l2_normalize/Sum:output:06tf.math.l2_normalize_4/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:����������
)tf.math.l2_normalize_4/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_4/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
#tf.math.l2_normalize_4/l2_normalizeMul&sequential_4/dense_19/BiasAdd:output:0-tf.math.l2_normalize_4/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@�
*tf.math.l2_normalize_5/l2_normalize/SquareSquare&sequential_5/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:���������@{
9tf.math.l2_normalize_5/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
'tf.math.l2_normalize_5/l2_normalize/SumSum.tf.math.l2_normalize_5/l2_normalize/Square:y:0Btf.math.l2_normalize_5/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(r
-tf.math.l2_normalize_5/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
+tf.math.l2_normalize_5/l2_normalize/MaximumMaximum0tf.math.l2_normalize_5/l2_normalize/Sum:output:06tf.math.l2_normalize_5/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:����������
)tf.math.l2_normalize_5/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_5/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
#tf.math.l2_normalize_5/l2_normalizeMul&sequential_5/dense_23/BiasAdd:output:0-tf.math.l2_normalize_5/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@V
dot_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot_2/ExpandDims
ExpandDims'tf.math.l2_normalize_4/l2_normalize:z:0dot_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������@X
dot_2/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot_2/ExpandDims_1
ExpandDims'tf.math.l2_normalize_5/l2_normalize:z:0dot_2/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:���������@�
dot_2/MatMulBatchMatMulV2dot_2/ExpandDims:output:0dot_2/ExpandDims_1:output:0*
T0*+
_output_shapes
:���������^
dot_2/ShapeShapedot_2/MatMul:output:0*
T0*
_output_shapes
::��x
dot_2/SqueezeSqueezedot_2/MatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
e
IdentityIdentitydot_2/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp-^sequential_4/dense_16/BiasAdd/ReadVariableOp,^sequential_4/dense_16/MatMul/ReadVariableOp-^sequential_4/dense_17/BiasAdd/ReadVariableOp,^sequential_4/dense_17/MatMul/ReadVariableOp-^sequential_4/dense_18/BiasAdd/ReadVariableOp,^sequential_4/dense_18/MatMul/ReadVariableOp-^sequential_4/dense_19/BiasAdd/ReadVariableOp,^sequential_4/dense_19/MatMul/ReadVariableOp-^sequential_5/dense_20/BiasAdd/ReadVariableOp,^sequential_5/dense_20/MatMul/ReadVariableOp-^sequential_5/dense_21/BiasAdd/ReadVariableOp,^sequential_5/dense_21/MatMul/ReadVariableOp-^sequential_5/dense_22/BiasAdd/ReadVariableOp,^sequential_5/dense_22/MatMul/ReadVariableOp-^sequential_5/dense_23/BiasAdd/ReadVariableOp,^sequential_5/dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:���������:���������: : : : : : : : : : : : : : : : 2\
,sequential_4/dense_16/BiasAdd/ReadVariableOp,sequential_4/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_16/MatMul/ReadVariableOp+sequential_4/dense_16/MatMul/ReadVariableOp2\
,sequential_4/dense_17/BiasAdd/ReadVariableOp,sequential_4/dense_17/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_17/MatMul/ReadVariableOp+sequential_4/dense_17/MatMul/ReadVariableOp2\
,sequential_4/dense_18/BiasAdd/ReadVariableOp,sequential_4/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_18/MatMul/ReadVariableOp+sequential_4/dense_18/MatMul/ReadVariableOp2\
,sequential_4/dense_19/BiasAdd/ReadVariableOp,sequential_4/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_19/MatMul/ReadVariableOp+sequential_4/dense_19/MatMul/ReadVariableOp2\
,sequential_5/dense_20/BiasAdd/ReadVariableOp,sequential_5/dense_20/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_20/MatMul/ReadVariableOp+sequential_5/dense_20/MatMul/ReadVariableOp2\
,sequential_5/dense_21/BiasAdd/ReadVariableOp,sequential_5/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_21/MatMul/ReadVariableOp+sequential_5/dense_21/MatMul/ReadVariableOp2\
,sequential_5/dense_22/BiasAdd/ReadVariableOp,sequential_5/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_22/MatMul/ReadVariableOp+sequential_5/dense_22/MatMul/ReadVariableOp2\
,sequential_5/dense_23/BiasAdd/ReadVariableOp,sequential_5/dense_23/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_23/MatMul/ReadVariableOp+sequential_5/dense_23/MatMul/ReadVariableOp:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0
�%
�
I__inference_sequential_5_layer_call_and_return_conditional_losses_1204414

inputs:
'dense_20_matmul_readvariableop_resource:	�7
(dense_20_biasadd_readvariableop_resource:	�;
'dense_21_matmul_readvariableop_resource:
��7
(dense_21_biasadd_readvariableop_resource:	�;
'dense_22_matmul_readvariableop_resource:
��7
(dense_22_biasadd_readvariableop_resource:	�:
'dense_23_matmul_readvariableop_resource:	�@6
(dense_23_biasadd_readvariableop_resource:@
identity��dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0|
dense_20/MatMulMatMulinputs&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*(
_output_shapes
:����������n
dropout_4/IdentityIdentitydense_20/Relu:activations:0*
T0*(
_output_shapes
:�����������
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_21/MatMulMatMuldropout_4/Identity:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*(
_output_shapes
:����������n
dropout_5/IdentityIdentitydense_21/Relu:activations:0*
T0*(
_output_shapes
:�����������
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_22/MatMulMatMuldropout_5/Identity:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_23/MatMulMatMuldense_22/Relu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@h
IdentityIdentitydense_23/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�*
�
D__inference_model_2_layer_call_and_return_conditional_losses_1203588
inf_feature
own_feature'
sequential_5_1203538:	�#
sequential_5_1203540:	�(
sequential_5_1203542:
��#
sequential_5_1203544:	�(
sequential_5_1203546:
��#
sequential_5_1203548:	�'
sequential_5_1203550:	�@"
sequential_5_1203552:@'
sequential_4_1203555:	�#
sequential_4_1203557:	�(
sequential_4_1203559:
��#
sequential_4_1203561:	�'
sequential_4_1203563:	�@"
sequential_4_1203565:@&
sequential_4_1203567:@@"
sequential_4_1203569:@
identity��$sequential_4/StatefulPartitionedCall�$sequential_5/StatefulPartitionedCall�
$sequential_5/StatefulPartitionedCallStatefulPartitionedCallown_featuresequential_5_1203538sequential_5_1203540sequential_5_1203542sequential_5_1203544sequential_5_1203546sequential_5_1203548sequential_5_1203550sequential_5_1203552*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_1203343�
$sequential_4/StatefulPartitionedCallStatefulPartitionedCallinf_featuresequential_4_1203555sequential_4_1203557sequential_4_1203559sequential_4_1203561sequential_4_1203563sequential_4_1203565sequential_4_1203567sequential_4_1203569*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_1203008�
*tf.math.l2_normalize_4/l2_normalize/SquareSquare-sequential_4/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������@{
9tf.math.l2_normalize_4/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
'tf.math.l2_normalize_4/l2_normalize/SumSum.tf.math.l2_normalize_4/l2_normalize/Square:y:0Btf.math.l2_normalize_4/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(r
-tf.math.l2_normalize_4/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
+tf.math.l2_normalize_4/l2_normalize/MaximumMaximum0tf.math.l2_normalize_4/l2_normalize/Sum:output:06tf.math.l2_normalize_4/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:����������
)tf.math.l2_normalize_4/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_4/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
#tf.math.l2_normalize_4/l2_normalizeMul-sequential_4/StatefulPartitionedCall:output:0-tf.math.l2_normalize_4/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@�
*tf.math.l2_normalize_5/l2_normalize/SquareSquare-sequential_5/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������@{
9tf.math.l2_normalize_5/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
'tf.math.l2_normalize_5/l2_normalize/SumSum.tf.math.l2_normalize_5/l2_normalize/Square:y:0Btf.math.l2_normalize_5/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(r
-tf.math.l2_normalize_5/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
+tf.math.l2_normalize_5/l2_normalize/MaximumMaximum0tf.math.l2_normalize_5/l2_normalize/Sum:output:06tf.math.l2_normalize_5/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:����������
)tf.math.l2_normalize_5/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_5/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
#tf.math.l2_normalize_5/l2_normalizeMul-sequential_5/StatefulPartitionedCall:output:0-tf.math.l2_normalize_5/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@�
dot_2/PartitionedCallPartitionedCall'tf.math.l2_normalize_4/l2_normalize:z:0'tf.math.l2_normalize_5/l2_normalize:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dot_2_layer_call_and_return_conditional_losses_1203531m
IdentityIdentitydot_2/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:���������:���������: : : : : : : : : : : : : : : : 2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall:TP
'
_output_shapes
:���������
%
_user_specified_nameown_feature:T P
'
_output_shapes
:���������
%
_user_specified_nameinf_feature
�
�
*__inference_dense_22_layer_call_fn_1204668

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_1203208p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
.__inference_sequential_4_layer_call_fn_1204212

inputs
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_1203008o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)__inference_model_2_layer_call_fn_1203974
inputs_0
inputs_1
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�@
	unknown_6:@
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�@

unknown_12:@

unknown_13:@@

unknown_14:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_1203738o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:���������:���������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0
�
G
+__inference_dropout_2_layer_call_fn_1204462

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_1202908a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�2
 __inference__traced_save_1205058
file_prefix9
&read_disablecopyonread_dense_16_kernel:	�5
&read_1_disablecopyonread_dense_16_bias:	�<
(read_2_disablecopyonread_dense_17_kernel:
��5
&read_3_disablecopyonread_dense_17_bias:	�;
(read_4_disablecopyonread_dense_18_kernel:	�@4
&read_5_disablecopyonread_dense_18_bias:@:
(read_6_disablecopyonread_dense_19_kernel:@@4
&read_7_disablecopyonread_dense_19_bias:@;
(read_8_disablecopyonread_dense_20_kernel:	�5
&read_9_disablecopyonread_dense_20_bias:	�=
)read_10_disablecopyonread_dense_21_kernel:
��6
'read_11_disablecopyonread_dense_21_bias:	�=
)read_12_disablecopyonread_dense_22_kernel:
��6
'read_13_disablecopyonread_dense_22_bias:	�<
)read_14_disablecopyonread_dense_23_kernel:	�@5
'read_15_disablecopyonread_dense_23_bias:@-
#read_16_disablecopyonread_iteration:	 1
'read_17_disablecopyonread_learning_rate: C
0read_18_disablecopyonread_adam_m_dense_16_kernel:	�C
0read_19_disablecopyonread_adam_v_dense_16_kernel:	�=
.read_20_disablecopyonread_adam_m_dense_16_bias:	�=
.read_21_disablecopyonread_adam_v_dense_16_bias:	�D
0read_22_disablecopyonread_adam_m_dense_17_kernel:
��D
0read_23_disablecopyonread_adam_v_dense_17_kernel:
��=
.read_24_disablecopyonread_adam_m_dense_17_bias:	�=
.read_25_disablecopyonread_adam_v_dense_17_bias:	�C
0read_26_disablecopyonread_adam_m_dense_18_kernel:	�@C
0read_27_disablecopyonread_adam_v_dense_18_kernel:	�@<
.read_28_disablecopyonread_adam_m_dense_18_bias:@<
.read_29_disablecopyonread_adam_v_dense_18_bias:@B
0read_30_disablecopyonread_adam_m_dense_19_kernel:@@B
0read_31_disablecopyonread_adam_v_dense_19_kernel:@@<
.read_32_disablecopyonread_adam_m_dense_19_bias:@<
.read_33_disablecopyonread_adam_v_dense_19_bias:@C
0read_34_disablecopyonread_adam_m_dense_20_kernel:	�C
0read_35_disablecopyonread_adam_v_dense_20_kernel:	�=
.read_36_disablecopyonread_adam_m_dense_20_bias:	�=
.read_37_disablecopyonread_adam_v_dense_20_bias:	�D
0read_38_disablecopyonread_adam_m_dense_21_kernel:
��D
0read_39_disablecopyonread_adam_v_dense_21_kernel:
��=
.read_40_disablecopyonread_adam_m_dense_21_bias:	�=
.read_41_disablecopyonread_adam_v_dense_21_bias:	�D
0read_42_disablecopyonread_adam_m_dense_22_kernel:
��D
0read_43_disablecopyonread_adam_v_dense_22_kernel:
��=
.read_44_disablecopyonread_adam_m_dense_22_bias:	�=
.read_45_disablecopyonread_adam_v_dense_22_bias:	�C
0read_46_disablecopyonread_adam_m_dense_23_kernel:	�@C
0read_47_disablecopyonread_adam_v_dense_23_kernel:	�@<
.read_48_disablecopyonread_adam_m_dense_23_bias:@<
.read_49_disablecopyonread_adam_v_dense_23_bias:@+
!read_50_disablecopyonread_total_2: +
!read_51_disablecopyonread_count_2: +
!read_52_disablecopyonread_total_1: +
!read_53_disablecopyonread_count_1: )
read_54_disablecopyonread_total: )
read_55_disablecopyonread_count: 
savev2_const
identity_113��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_dense_16_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_dense_16_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	�z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_dense_16_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_dense_16_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_dense_17_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_dense_17_kernel^Read_2/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_dense_17_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_dense_17_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_dense_18_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_dense_18_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0n

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@d

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_dense_18_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_dense_18_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:@|
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_dense_19_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_dense_19_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:@@z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_dense_19_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_dense_19_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:@|
Read_8/DisableCopyOnReadDisableCopyOnRead(read_8_disablecopyonread_dense_20_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp(read_8_disablecopyonread_dense_20_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0o
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:	�z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_dense_20_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_dense_20_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_10/DisableCopyOnReadDisableCopyOnRead)read_10_disablecopyonread_dense_21_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp)read_10_disablecopyonread_dense_21_kernel^Read_10/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��|
Read_11/DisableCopyOnReadDisableCopyOnRead'read_11_disablecopyonread_dense_21_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp'read_11_disablecopyonread_dense_21_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_12/DisableCopyOnReadDisableCopyOnRead)read_12_disablecopyonread_dense_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp)read_12_disablecopyonread_dense_22_kernel^Read_12/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��|
Read_13/DisableCopyOnReadDisableCopyOnRead'read_13_disablecopyonread_dense_22_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp'read_13_disablecopyonread_dense_22_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_14/DisableCopyOnReadDisableCopyOnRead)read_14_disablecopyonread_dense_23_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp)read_14_disablecopyonread_dense_23_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@|
Read_15/DisableCopyOnReadDisableCopyOnRead'read_15_disablecopyonread_dense_23_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp'read_15_disablecopyonread_dense_23_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:@x
Read_16/DisableCopyOnReadDisableCopyOnRead#read_16_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp#read_16_disablecopyonread_iteration^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_17/DisableCopyOnReadDisableCopyOnRead'read_17_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp'read_17_disablecopyonread_learning_rate^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_18/DisableCopyOnReadDisableCopyOnRead0read_18_disablecopyonread_adam_m_dense_16_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp0read_18_disablecopyonread_adam_m_dense_16_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_19/DisableCopyOnReadDisableCopyOnRead0read_19_disablecopyonread_adam_v_dense_16_kernel"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp0read_19_disablecopyonread_adam_v_dense_16_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_20/DisableCopyOnReadDisableCopyOnRead.read_20_disablecopyonread_adam_m_dense_16_bias"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp.read_20_disablecopyonread_adam_m_dense_16_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_21/DisableCopyOnReadDisableCopyOnRead.read_21_disablecopyonread_adam_v_dense_16_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp.read_21_disablecopyonread_adam_v_dense_16_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_22/DisableCopyOnReadDisableCopyOnRead0read_22_disablecopyonread_adam_m_dense_17_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp0read_22_disablecopyonread_adam_m_dense_17_kernel^Read_22/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_23/DisableCopyOnReadDisableCopyOnRead0read_23_disablecopyonread_adam_v_dense_17_kernel"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp0read_23_disablecopyonread_adam_v_dense_17_kernel^Read_23/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_24/DisableCopyOnReadDisableCopyOnRead.read_24_disablecopyonread_adam_m_dense_17_bias"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp.read_24_disablecopyonread_adam_m_dense_17_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_25/DisableCopyOnReadDisableCopyOnRead.read_25_disablecopyonread_adam_v_dense_17_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp.read_25_disablecopyonread_adam_v_dense_17_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_26/DisableCopyOnReadDisableCopyOnRead0read_26_disablecopyonread_adam_m_dense_18_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp0read_26_disablecopyonread_adam_m_dense_18_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@�
Read_27/DisableCopyOnReadDisableCopyOnRead0read_27_disablecopyonread_adam_v_dense_18_kernel"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp0read_27_disablecopyonread_adam_v_dense_18_kernel^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@�
Read_28/DisableCopyOnReadDisableCopyOnRead.read_28_disablecopyonread_adam_m_dense_18_bias"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp.read_28_disablecopyonread_adam_m_dense_18_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_29/DisableCopyOnReadDisableCopyOnRead.read_29_disablecopyonread_adam_v_dense_18_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp.read_29_disablecopyonread_adam_v_dense_18_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_30/DisableCopyOnReadDisableCopyOnRead0read_30_disablecopyonread_adam_m_dense_19_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp0read_30_disablecopyonread_adam_m_dense_19_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes

:@@�
Read_31/DisableCopyOnReadDisableCopyOnRead0read_31_disablecopyonread_adam_v_dense_19_kernel"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp0read_31_disablecopyonread_adam_v_dense_19_kernel^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes

:@@�
Read_32/DisableCopyOnReadDisableCopyOnRead.read_32_disablecopyonread_adam_m_dense_19_bias"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp.read_32_disablecopyonread_adam_m_dense_19_bias^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_33/DisableCopyOnReadDisableCopyOnRead.read_33_disablecopyonread_adam_v_dense_19_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp.read_33_disablecopyonread_adam_v_dense_19_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_34/DisableCopyOnReadDisableCopyOnRead0read_34_disablecopyonread_adam_m_dense_20_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp0read_34_disablecopyonread_adam_m_dense_20_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_35/DisableCopyOnReadDisableCopyOnRead0read_35_disablecopyonread_adam_v_dense_20_kernel"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp0read_35_disablecopyonread_adam_v_dense_20_kernel^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_36/DisableCopyOnReadDisableCopyOnRead.read_36_disablecopyonread_adam_m_dense_20_bias"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp.read_36_disablecopyonread_adam_m_dense_20_bias^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_37/DisableCopyOnReadDisableCopyOnRead.read_37_disablecopyonread_adam_v_dense_20_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp.read_37_disablecopyonread_adam_v_dense_20_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_38/DisableCopyOnReadDisableCopyOnRead0read_38_disablecopyonread_adam_m_dense_21_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp0read_38_disablecopyonread_adam_m_dense_21_kernel^Read_38/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_39/DisableCopyOnReadDisableCopyOnRead0read_39_disablecopyonread_adam_v_dense_21_kernel"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp0read_39_disablecopyonread_adam_v_dense_21_kernel^Read_39/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_40/DisableCopyOnReadDisableCopyOnRead.read_40_disablecopyonread_adam_m_dense_21_bias"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp.read_40_disablecopyonread_adam_m_dense_21_bias^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_41/DisableCopyOnReadDisableCopyOnRead.read_41_disablecopyonread_adam_v_dense_21_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp.read_41_disablecopyonread_adam_v_dense_21_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_42/DisableCopyOnReadDisableCopyOnRead0read_42_disablecopyonread_adam_m_dense_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp0read_42_disablecopyonread_adam_m_dense_22_kernel^Read_42/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_43/DisableCopyOnReadDisableCopyOnRead0read_43_disablecopyonread_adam_v_dense_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp0read_43_disablecopyonread_adam_v_dense_22_kernel^Read_43/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_44/DisableCopyOnReadDisableCopyOnRead.read_44_disablecopyonread_adam_m_dense_22_bias"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp.read_44_disablecopyonread_adam_m_dense_22_bias^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_45/DisableCopyOnReadDisableCopyOnRead.read_45_disablecopyonread_adam_v_dense_22_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp.read_45_disablecopyonread_adam_v_dense_22_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_46/DisableCopyOnReadDisableCopyOnRead0read_46_disablecopyonread_adam_m_dense_23_kernel"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp0read_46_disablecopyonread_adam_m_dense_23_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@�
Read_47/DisableCopyOnReadDisableCopyOnRead0read_47_disablecopyonread_adam_v_dense_23_kernel"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp0read_47_disablecopyonread_adam_v_dense_23_kernel^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@�
Read_48/DisableCopyOnReadDisableCopyOnRead.read_48_disablecopyonread_adam_m_dense_23_bias"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp.read_48_disablecopyonread_adam_m_dense_23_bias^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_49/DisableCopyOnReadDisableCopyOnRead.read_49_disablecopyonread_adam_v_dense_23_bias"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp.read_49_disablecopyonread_adam_v_dense_23_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
:@v
Read_50/DisableCopyOnReadDisableCopyOnRead!read_50_disablecopyonread_total_2"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp!read_50_disablecopyonread_total_2^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_51/DisableCopyOnReadDisableCopyOnRead!read_51_disablecopyonread_count_2"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp!read_51_disablecopyonread_count_2^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_52/DisableCopyOnReadDisableCopyOnRead!read_52_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp!read_52_disablecopyonread_total_1^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_53/DisableCopyOnReadDisableCopyOnRead!read_53_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp!read_53_disablecopyonread_count_1^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_54/DisableCopyOnReadDisableCopyOnReadread_54_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOpread_54_disablecopyonread_total^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_55/DisableCopyOnReadDisableCopyOnReadread_55_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOpread_55_disablecopyonread_count^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*�
value�B�9B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*�
value|Bz9B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *G
dtypes=
;29	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_112Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_113IdentityIdentity_112:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "%
identity_113Identity_113:output:0*�
_input_shapesv
t: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:9

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
)__inference_model_2_layer_call_fn_1203936
inputs_0
inputs_1
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�@
	unknown_6:@
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�@

unknown_12:@

unknown_13:@@

unknown_14:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_1203646o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:���������:���������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0
�
�
*__inference_dense_19_layer_call_fn_1204555

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_19_layer_call_and_return_conditional_losses_1202889o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
E__inference_dense_20_layer_call_and_return_conditional_losses_1204585

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

e
F__inference_dropout_2_layer_call_and_return_conditional_losses_1204474

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   Ae
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_16_layer_call_and_return_conditional_losses_1204452

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_dense_16_layer_call_fn_1204441

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_1202811p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
d
F__inference_dropout_4_layer_call_and_return_conditional_losses_1203243

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_16_layer_call_and_return_conditional_losses_1202811

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
S
'__inference_dot_2_layer_call_fn_1204420
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dot_2_layer_call_and_return_conditional_losses_1203531`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:QM
'
_output_shapes
:���������@
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs_0
�
�
I__inference_sequential_5_layer_call_and_return_conditional_losses_1203343

inputs#
dense_20_1203320:	�
dense_20_1203322:	�$
dense_21_1203326:
��
dense_21_1203328:	�$
dense_22_1203332:
��
dense_22_1203334:	�#
dense_23_1203337:	�@
dense_23_1203339:@
identity�� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCallinputsdense_20_1203320dense_20_1203322*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1203146�
dropout_4/PartitionedCallPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_4_layer_call_and_return_conditional_losses_1203243�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_21_1203326dense_21_1203328*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1203177�
dropout_5/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_1203254�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_22_1203332dense_22_1203334*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_1203208�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_1203337dense_23_1203339*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_1203224x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_1204479

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_18_layer_call_and_return_conditional_losses_1202873

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

e
F__inference_dropout_3_layer_call_and_return_conditional_losses_1202860

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   Ae
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�)
�
D__inference_model_2_layer_call_and_return_conditional_losses_1203646

inputs
inputs_1'
sequential_5_1203596:	�#
sequential_5_1203598:	�(
sequential_5_1203600:
��#
sequential_5_1203602:	�(
sequential_5_1203604:
��#
sequential_5_1203606:	�'
sequential_5_1203608:	�@"
sequential_5_1203610:@'
sequential_4_1203613:	�#
sequential_4_1203615:	�(
sequential_4_1203617:
��#
sequential_4_1203619:	�'
sequential_4_1203621:	�@"
sequential_4_1203623:@&
sequential_4_1203625:@@"
sequential_4_1203627:@
identity��$sequential_4/StatefulPartitionedCall�$sequential_5/StatefulPartitionedCall�
$sequential_5/StatefulPartitionedCallStatefulPartitionedCallinputs_1sequential_5_1203596sequential_5_1203598sequential_5_1203600sequential_5_1203602sequential_5_1203604sequential_5_1203606sequential_5_1203608sequential_5_1203610*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_1203296�
$sequential_4/StatefulPartitionedCallStatefulPartitionedCallinputssequential_4_1203613sequential_4_1203615sequential_4_1203617sequential_4_1203619sequential_4_1203621sequential_4_1203623sequential_4_1203625sequential_4_1203627*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_1202961�
*tf.math.l2_normalize_4/l2_normalize/SquareSquare-sequential_4/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������@{
9tf.math.l2_normalize_4/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
'tf.math.l2_normalize_4/l2_normalize/SumSum.tf.math.l2_normalize_4/l2_normalize/Square:y:0Btf.math.l2_normalize_4/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(r
-tf.math.l2_normalize_4/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
+tf.math.l2_normalize_4/l2_normalize/MaximumMaximum0tf.math.l2_normalize_4/l2_normalize/Sum:output:06tf.math.l2_normalize_4/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:����������
)tf.math.l2_normalize_4/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_4/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
#tf.math.l2_normalize_4/l2_normalizeMul-sequential_4/StatefulPartitionedCall:output:0-tf.math.l2_normalize_4/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@�
*tf.math.l2_normalize_5/l2_normalize/SquareSquare-sequential_5/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������@{
9tf.math.l2_normalize_5/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
'tf.math.l2_normalize_5/l2_normalize/SumSum.tf.math.l2_normalize_5/l2_normalize/Square:y:0Btf.math.l2_normalize_5/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(r
-tf.math.l2_normalize_5/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
+tf.math.l2_normalize_5/l2_normalize/MaximumMaximum0tf.math.l2_normalize_5/l2_normalize/Sum:output:06tf.math.l2_normalize_5/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:����������
)tf.math.l2_normalize_5/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_5/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
#tf.math.l2_normalize_5/l2_normalizeMul-sequential_5/StatefulPartitionedCall:output:0-tf.math.l2_normalize_5/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@�
dot_2/PartitionedCallPartitionedCall'tf.math.l2_normalize_4/l2_normalize:z:0'tf.math.l2_normalize_5/l2_normalize:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dot_2_layer_call_and_return_conditional_losses_1203531m
IdentityIdentitydot_2/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:���������:���������: : : : : : : : : : : : : : : : 2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�)
�
D__inference_model_2_layer_call_and_return_conditional_losses_1203738

inputs
inputs_1'
sequential_5_1203688:	�#
sequential_5_1203690:	�(
sequential_5_1203692:
��#
sequential_5_1203694:	�(
sequential_5_1203696:
��#
sequential_5_1203698:	�'
sequential_5_1203700:	�@"
sequential_5_1203702:@'
sequential_4_1203705:	�#
sequential_4_1203707:	�(
sequential_4_1203709:
��#
sequential_4_1203711:	�'
sequential_4_1203713:	�@"
sequential_4_1203715:@&
sequential_4_1203717:@@"
sequential_4_1203719:@
identity��$sequential_4/StatefulPartitionedCall�$sequential_5/StatefulPartitionedCall�
$sequential_5/StatefulPartitionedCallStatefulPartitionedCallinputs_1sequential_5_1203688sequential_5_1203690sequential_5_1203692sequential_5_1203694sequential_5_1203696sequential_5_1203698sequential_5_1203700sequential_5_1203702*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_1203343�
$sequential_4/StatefulPartitionedCallStatefulPartitionedCallinputssequential_4_1203705sequential_4_1203707sequential_4_1203709sequential_4_1203711sequential_4_1203713sequential_4_1203715sequential_4_1203717sequential_4_1203719*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_1203008�
*tf.math.l2_normalize_4/l2_normalize/SquareSquare-sequential_4/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������@{
9tf.math.l2_normalize_4/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
'tf.math.l2_normalize_4/l2_normalize/SumSum.tf.math.l2_normalize_4/l2_normalize/Square:y:0Btf.math.l2_normalize_4/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(r
-tf.math.l2_normalize_4/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
+tf.math.l2_normalize_4/l2_normalize/MaximumMaximum0tf.math.l2_normalize_4/l2_normalize/Sum:output:06tf.math.l2_normalize_4/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:����������
)tf.math.l2_normalize_4/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_4/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
#tf.math.l2_normalize_4/l2_normalizeMul-sequential_4/StatefulPartitionedCall:output:0-tf.math.l2_normalize_4/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@�
*tf.math.l2_normalize_5/l2_normalize/SquareSquare-sequential_5/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������@{
9tf.math.l2_normalize_5/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
'tf.math.l2_normalize_5/l2_normalize/SumSum.tf.math.l2_normalize_5/l2_normalize/Square:y:0Btf.math.l2_normalize_5/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(r
-tf.math.l2_normalize_5/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
+tf.math.l2_normalize_5/l2_normalize/MaximumMaximum0tf.math.l2_normalize_5/l2_normalize/Sum:output:06tf.math.l2_normalize_5/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:����������
)tf.math.l2_normalize_5/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_5/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
#tf.math.l2_normalize_5/l2_normalizeMul-sequential_5/StatefulPartitionedCall:output:0-tf.math.l2_normalize_5/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������@�
dot_2/PartitionedCallPartitionedCall'tf.math.l2_normalize_4/l2_normalize:z:0'tf.math.l2_normalize_5/l2_normalize:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dot_2_layer_call_and_return_conditional_losses_1203531m
IdentityIdentitydot_2/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:���������:���������: : : : : : : : : : : : : : : : 2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_1203898
inf_feature
own_feature
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�@
	unknown_6:@
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�@

unknown_12:@

unknown_13:@@

unknown_14:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinf_featureown_featureunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_1202796o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:���������:���������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:TP
'
_output_shapes
:���������
%
_user_specified_nameown_feature:T P
'
_output_shapes
:���������
%
_user_specified_nameinf_feature"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
inf_feature4
serving_default_inf_feature:0���������
C
own_feature4
serving_default_own_feature:0���������9
dot_20
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
 layer-3
!layer_with_weights-2
!layer-4
"layer_with_weights-3
"layer-5
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_sequential
(
)	keras_api"
_tf_keras_layer
(
*	keras_api"
_tf_keras_layer
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
�
10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13
?14
@15"
trackable_list_wrapper
�
10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13
?14
@15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ftrace_0
Gtrace_1
Htrace_2
Itrace_32�
)__inference_model_2_layer_call_fn_1203681
)__inference_model_2_layer_call_fn_1203773
)__inference_model_2_layer_call_fn_1203936
)__inference_model_2_layer_call_fn_1203974�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zFtrace_0zGtrace_1zHtrace_2zItrace_3
�
Jtrace_0
Ktrace_1
Ltrace_2
Mtrace_32�
D__inference_model_2_layer_call_and_return_conditional_losses_1203534
D__inference_model_2_layer_call_and_return_conditional_losses_1203588
D__inference_model_2_layer_call_and_return_conditional_losses_1204086
D__inference_model_2_layer_call_and_return_conditional_losses_1204170�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zJtrace_0zKtrace_1zLtrace_2zMtrace_3
�B�
"__inference__wrapped_model_1202796inf_featureown_feature"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
N
_variables
O_iterations
P_learning_rate
Q_index_dict
R
_momentums
S_velocities
T_update_step_xla"
experimentalOptimizer
,
Userving_default"
signature_map
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

1kernel
2bias"
_tf_keras_layer
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses
b_random_generator"
_tf_keras_layer
�
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

3kernel
4bias"
_tf_keras_layer
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses
o_random_generator"
_tf_keras_layer
�
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses

5kernel
6bias"
_tf_keras_layer
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses

7kernel
8bias"
_tf_keras_layer
X
10
21
32
43
54
65
76
87"
trackable_list_wrapper
X
10
21
32
43
54
65
76
87"
trackable_list_wrapper
 "
trackable_list_wrapper
�
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
.__inference_sequential_4_layer_call_fn_1202980
.__inference_sequential_4_layer_call_fn_1203027
.__inference_sequential_4_layer_call_fn_1204191
.__inference_sequential_4_layer_call_fn_1204212�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
I__inference_sequential_4_layer_call_and_return_conditional_losses_1202896
I__inference_sequential_4_layer_call_and_return_conditional_losses_1202932
I__inference_sequential_4_layer_call_and_return_conditional_losses_1204259
I__inference_sequential_4_layer_call_and_return_conditional_losses_1204292�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

9kernel
:bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

;kernel
<bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

=kernel
>bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

?kernel
@bias"
_tf_keras_layer
X
90
:1
;2
<3
=4
>5
?6
@7"
trackable_list_wrapper
X
90
:1
;2
<3
=4
>5
?6
@7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
.__inference_sequential_5_layer_call_fn_1203315
.__inference_sequential_5_layer_call_fn_1203362
.__inference_sequential_5_layer_call_fn_1204313
.__inference_sequential_5_layer_call_fn_1204334�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
I__inference_sequential_5_layer_call_and_return_conditional_losses_1203231
I__inference_sequential_5_layer_call_and_return_conditional_losses_1203267
I__inference_sequential_5_layer_call_and_return_conditional_losses_1204381
I__inference_sequential_5_layer_call_and_return_conditional_losses_1204414�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_dot_2_layer_call_fn_1204420�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_dot_2_layer_call_and_return_conditional_losses_1204432�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 	�2dense_16/kernel
:�2dense_16/bias
#:!
��2dense_17/kernel
:�2dense_17/bias
": 	�@2dense_18/kernel
:@2dense_18/bias
!:@@2dense_19/kernel
:@2dense_19/bias
": 	�2dense_20/kernel
:�2dense_20/bias
#:!
��2dense_21/kernel
:�2dense_21/bias
#:!
��2dense_22/kernel
:�2dense_22/bias
": 	�@2dense_23/kernel
:@2dense_23/bias
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_model_2_layer_call_fn_1203681inf_featureown_feature"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_model_2_layer_call_fn_1203773inf_featureown_feature"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_model_2_layer_call_fn_1203936inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_model_2_layer_call_fn_1203974inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_model_2_layer_call_and_return_conditional_losses_1203534inf_featureown_feature"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_model_2_layer_call_and_return_conditional_losses_1203588inf_featureown_feature"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_model_2_layer_call_and_return_conditional_losses_1204086inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_model_2_layer_call_and_return_conditional_losses_1204170inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
O0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
%__inference_signature_wrapper_1203898inf_featureown_feature"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_16_layer_call_fn_1204441�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_16_layer_call_and_return_conditional_losses_1204452�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_dropout_2_layer_call_fn_1204457
+__inference_dropout_2_layer_call_fn_1204462�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_dropout_2_layer_call_and_return_conditional_losses_1204474
F__inference_dropout_2_layer_call_and_return_conditional_losses_1204479�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_17_layer_call_fn_1204488�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_17_layer_call_and_return_conditional_losses_1204499�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_dropout_3_layer_call_fn_1204504
+__inference_dropout_3_layer_call_fn_1204509�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_dropout_3_layer_call_and_return_conditional_losses_1204521
F__inference_dropout_3_layer_call_and_return_conditional_losses_1204526�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_18_layer_call_fn_1204535�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_18_layer_call_and_return_conditional_losses_1204546�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_19_layer_call_fn_1204555�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_19_layer_call_and_return_conditional_losses_1204565�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_sequential_4_layer_call_fn_1202980dense_16_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_4_layer_call_fn_1203027dense_16_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_4_layer_call_fn_1204191inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_4_layer_call_fn_1204212inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_4_layer_call_and_return_conditional_losses_1202896dense_16_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_4_layer_call_and_return_conditional_losses_1202932dense_16_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_4_layer_call_and_return_conditional_losses_1204259inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_4_layer_call_and_return_conditional_losses_1204292inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_20_layer_call_fn_1204574�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_20_layer_call_and_return_conditional_losses_1204585�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_dropout_4_layer_call_fn_1204590
+__inference_dropout_4_layer_call_fn_1204595�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_dropout_4_layer_call_and_return_conditional_losses_1204607
F__inference_dropout_4_layer_call_and_return_conditional_losses_1204612�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_21_layer_call_fn_1204621�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_21_layer_call_and_return_conditional_losses_1204632�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_dropout_5_layer_call_fn_1204637
+__inference_dropout_5_layer_call_fn_1204642�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_dropout_5_layer_call_and_return_conditional_losses_1204654
F__inference_dropout_5_layer_call_and_return_conditional_losses_1204659�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_22_layer_call_fn_1204668�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_22_layer_call_and_return_conditional_losses_1204679�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_23_layer_call_fn_1204688�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_23_layer_call_and_return_conditional_losses_1204698�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
J
0
1
2
 3
!4
"5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_sequential_5_layer_call_fn_1203315dense_20_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_5_layer_call_fn_1203362dense_20_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_5_layer_call_fn_1204313inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_5_layer_call_fn_1204334inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_5_layer_call_and_return_conditional_losses_1203231dense_20_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_5_layer_call_and_return_conditional_losses_1203267dense_20_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_5_layer_call_and_return_conditional_losses_1204381inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_5_layer_call_and_return_conditional_losses_1204414inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
'__inference_dot_2_layer_call_fn_1204420inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dot_2_layer_call_and_return_conditional_losses_1204432inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
':%	�2Adam/m/dense_16/kernel
':%	�2Adam/v/dense_16/kernel
!:�2Adam/m/dense_16/bias
!:�2Adam/v/dense_16/bias
(:&
��2Adam/m/dense_17/kernel
(:&
��2Adam/v/dense_17/kernel
!:�2Adam/m/dense_17/bias
!:�2Adam/v/dense_17/bias
':%	�@2Adam/m/dense_18/kernel
':%	�@2Adam/v/dense_18/kernel
 :@2Adam/m/dense_18/bias
 :@2Adam/v/dense_18/bias
&:$@@2Adam/m/dense_19/kernel
&:$@@2Adam/v/dense_19/kernel
 :@2Adam/m/dense_19/bias
 :@2Adam/v/dense_19/bias
':%	�2Adam/m/dense_20/kernel
':%	�2Adam/v/dense_20/kernel
!:�2Adam/m/dense_20/bias
!:�2Adam/v/dense_20/bias
(:&
��2Adam/m/dense_21/kernel
(:&
��2Adam/v/dense_21/kernel
!:�2Adam/m/dense_21/bias
!:�2Adam/v/dense_21/bias
(:&
��2Adam/m/dense_22/kernel
(:&
��2Adam/v/dense_22/kernel
!:�2Adam/m/dense_22/bias
!:�2Adam/v/dense_22/bias
':%	�@2Adam/m/dense_23/kernel
':%	�@2Adam/v/dense_23/kernel
 :@2Adam/m/dense_23/bias
 :@2Adam/v/dense_23/bias
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
�B�
*__inference_dense_16_layer_call_fn_1204441inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_16_layer_call_and_return_conditional_losses_1204452inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_dropout_2_layer_call_fn_1204457inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_2_layer_call_fn_1204462inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_2_layer_call_and_return_conditional_losses_1204474inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_2_layer_call_and_return_conditional_losses_1204479inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_dense_17_layer_call_fn_1204488inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_17_layer_call_and_return_conditional_losses_1204499inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_dropout_3_layer_call_fn_1204504inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_3_layer_call_fn_1204509inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_3_layer_call_and_return_conditional_losses_1204521inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_3_layer_call_and_return_conditional_losses_1204526inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_dense_18_layer_call_fn_1204535inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_18_layer_call_and_return_conditional_losses_1204546inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_dense_19_layer_call_fn_1204555inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_19_layer_call_and_return_conditional_losses_1204565inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_dense_20_layer_call_fn_1204574inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_20_layer_call_and_return_conditional_losses_1204585inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_dropout_4_layer_call_fn_1204590inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_4_layer_call_fn_1204595inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_4_layer_call_and_return_conditional_losses_1204607inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_4_layer_call_and_return_conditional_losses_1204612inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_dense_21_layer_call_fn_1204621inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_21_layer_call_and_return_conditional_losses_1204632inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_dropout_5_layer_call_fn_1204637inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_5_layer_call_fn_1204642inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_5_layer_call_and_return_conditional_losses_1204654inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_5_layer_call_and_return_conditional_losses_1204659inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_dense_22_layer_call_fn_1204668inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_22_layer_call_and_return_conditional_losses_1204679inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_dense_23_layer_call_fn_1204688inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_23_layer_call_and_return_conditional_losses_1204698inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
"__inference__wrapped_model_1202796�9:;<=>?@12345678`�]
V�S
Q�N
%�"
inf_feature���������
%�"
own_feature���������
� "-�*
(
dot_2�
dot_2����������
E__inference_dense_16_layer_call_and_return_conditional_losses_1204452d12/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
*__inference_dense_16_layer_call_fn_1204441Y12/�,
%�"
 �
inputs���������
� ""�
unknown�����������
E__inference_dense_17_layer_call_and_return_conditional_losses_1204499e340�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
*__inference_dense_17_layer_call_fn_1204488Z340�-
&�#
!�
inputs����������
� ""�
unknown�����������
E__inference_dense_18_layer_call_and_return_conditional_losses_1204546d560�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������@
� �
*__inference_dense_18_layer_call_fn_1204535Y560�-
&�#
!�
inputs����������
� "!�
unknown���������@�
E__inference_dense_19_layer_call_and_return_conditional_losses_1204565c78/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������@
� �
*__inference_dense_19_layer_call_fn_1204555X78/�,
%�"
 �
inputs���������@
� "!�
unknown���������@�
E__inference_dense_20_layer_call_and_return_conditional_losses_1204585d9:/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
*__inference_dense_20_layer_call_fn_1204574Y9:/�,
%�"
 �
inputs���������
� ""�
unknown�����������
E__inference_dense_21_layer_call_and_return_conditional_losses_1204632e;<0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
*__inference_dense_21_layer_call_fn_1204621Z;<0�-
&�#
!�
inputs����������
� ""�
unknown�����������
E__inference_dense_22_layer_call_and_return_conditional_losses_1204679e=>0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
*__inference_dense_22_layer_call_fn_1204668Z=>0�-
&�#
!�
inputs����������
� ""�
unknown�����������
E__inference_dense_23_layer_call_and_return_conditional_losses_1204698d?@0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������@
� �
*__inference_dense_23_layer_call_fn_1204688Y?@0�-
&�#
!�
inputs����������
� "!�
unknown���������@�
B__inference_dot_2_layer_call_and_return_conditional_losses_1204432�Z�W
P�M
K�H
"�
inputs_0���������@
"�
inputs_1���������@
� ",�)
"�
tensor_0���������
� �
'__inference_dot_2_layer_call_fn_1204420Z�W
P�M
K�H
"�
inputs_0���������@
"�
inputs_1���������@
� "!�
unknown����������
F__inference_dropout_2_layer_call_and_return_conditional_losses_1204474e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
F__inference_dropout_2_layer_call_and_return_conditional_losses_1204479e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
+__inference_dropout_2_layer_call_fn_1204457Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
+__inference_dropout_2_layer_call_fn_1204462Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
F__inference_dropout_3_layer_call_and_return_conditional_losses_1204521e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
F__inference_dropout_3_layer_call_and_return_conditional_losses_1204526e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
+__inference_dropout_3_layer_call_fn_1204504Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
+__inference_dropout_3_layer_call_fn_1204509Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
F__inference_dropout_4_layer_call_and_return_conditional_losses_1204607e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
F__inference_dropout_4_layer_call_and_return_conditional_losses_1204612e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
+__inference_dropout_4_layer_call_fn_1204590Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
+__inference_dropout_4_layer_call_fn_1204595Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
F__inference_dropout_5_layer_call_and_return_conditional_losses_1204654e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
F__inference_dropout_5_layer_call_and_return_conditional_losses_1204659e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
+__inference_dropout_5_layer_call_fn_1204637Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
+__inference_dropout_5_layer_call_fn_1204642Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
D__inference_model_2_layer_call_and_return_conditional_losses_1203534�9:;<=>?@12345678h�e
^�[
Q�N
%�"
inf_feature���������
%�"
own_feature���������
p

 
� ",�)
"�
tensor_0���������
� �
D__inference_model_2_layer_call_and_return_conditional_losses_1203588�9:;<=>?@12345678h�e
^�[
Q�N
%�"
inf_feature���������
%�"
own_feature���������
p 

 
� ",�)
"�
tensor_0���������
� �
D__inference_model_2_layer_call_and_return_conditional_losses_1204086�9:;<=>?@12345678b�_
X�U
K�H
"�
inputs_0���������
"�
inputs_1���������
p

 
� ",�)
"�
tensor_0���������
� �
D__inference_model_2_layer_call_and_return_conditional_losses_1204170�9:;<=>?@12345678b�_
X�U
K�H
"�
inputs_0���������
"�
inputs_1���������
p 

 
� ",�)
"�
tensor_0���������
� �
)__inference_model_2_layer_call_fn_1203681�9:;<=>?@12345678h�e
^�[
Q�N
%�"
inf_feature���������
%�"
own_feature���������
p

 
� "!�
unknown����������
)__inference_model_2_layer_call_fn_1203773�9:;<=>?@12345678h�e
^�[
Q�N
%�"
inf_feature���������
%�"
own_feature���������
p 

 
� "!�
unknown����������
)__inference_model_2_layer_call_fn_1203936�9:;<=>?@12345678b�_
X�U
K�H
"�
inputs_0���������
"�
inputs_1���������
p

 
� "!�
unknown����������
)__inference_model_2_layer_call_fn_1203974�9:;<=>?@12345678b�_
X�U
K�H
"�
inputs_0���������
"�
inputs_1���������
p 

 
� "!�
unknown����������
I__inference_sequential_4_layer_call_and_return_conditional_losses_1202896y12345678?�<
5�2
(�%
dense_16_input���������
p

 
� ",�)
"�
tensor_0���������@
� �
I__inference_sequential_4_layer_call_and_return_conditional_losses_1202932y12345678?�<
5�2
(�%
dense_16_input���������
p 

 
� ",�)
"�
tensor_0���������@
� �
I__inference_sequential_4_layer_call_and_return_conditional_losses_1204259q123456787�4
-�*
 �
inputs���������
p

 
� ",�)
"�
tensor_0���������@
� �
I__inference_sequential_4_layer_call_and_return_conditional_losses_1204292q123456787�4
-�*
 �
inputs���������
p 

 
� ",�)
"�
tensor_0���������@
� �
.__inference_sequential_4_layer_call_fn_1202980n12345678?�<
5�2
(�%
dense_16_input���������
p

 
� "!�
unknown���������@�
.__inference_sequential_4_layer_call_fn_1203027n12345678?�<
5�2
(�%
dense_16_input���������
p 

 
� "!�
unknown���������@�
.__inference_sequential_4_layer_call_fn_1204191f123456787�4
-�*
 �
inputs���������
p

 
� "!�
unknown���������@�
.__inference_sequential_4_layer_call_fn_1204212f123456787�4
-�*
 �
inputs���������
p 

 
� "!�
unknown���������@�
I__inference_sequential_5_layer_call_and_return_conditional_losses_1203231y9:;<=>?@?�<
5�2
(�%
dense_20_input���������
p

 
� ",�)
"�
tensor_0���������@
� �
I__inference_sequential_5_layer_call_and_return_conditional_losses_1203267y9:;<=>?@?�<
5�2
(�%
dense_20_input���������
p 

 
� ",�)
"�
tensor_0���������@
� �
I__inference_sequential_5_layer_call_and_return_conditional_losses_1204381q9:;<=>?@7�4
-�*
 �
inputs���������
p

 
� ",�)
"�
tensor_0���������@
� �
I__inference_sequential_5_layer_call_and_return_conditional_losses_1204414q9:;<=>?@7�4
-�*
 �
inputs���������
p 

 
� ",�)
"�
tensor_0���������@
� �
.__inference_sequential_5_layer_call_fn_1203315n9:;<=>?@?�<
5�2
(�%
dense_20_input���������
p

 
� "!�
unknown���������@�
.__inference_sequential_5_layer_call_fn_1203362n9:;<=>?@?�<
5�2
(�%
dense_20_input���������
p 

 
� "!�
unknown���������@�
.__inference_sequential_5_layer_call_fn_1204313f9:;<=>?@7�4
-�*
 �
inputs���������
p

 
� "!�
unknown���������@�
.__inference_sequential_5_layer_call_fn_1204334f9:;<=>?@7�4
-�*
 �
inputs���������
p 

 
� "!�
unknown���������@�
%__inference_signature_wrapper_1203898�9:;<=>?@12345678y�v
� 
o�l
4
inf_feature%�"
inf_feature���������
4
own_feature%�"
own_feature���������"-�*
(
dot_2�
dot_2���������