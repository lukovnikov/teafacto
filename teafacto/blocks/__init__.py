# TODO: implement example dynamic data feed
# TODO: implement more objectives

# WARNING: no automatic param adding in Block.wrapply() --> might be source of bugs

# WARNING: using updates with theano.scan is not tested and might not work, actually, don't use updates

# TODO/TEST: parameter constraints

# TODO/TEST: regularization

# TODO/TEST: Val with all the rest

# TODO/TEST: data splitting tests

# IDEA: store parameter owners/block owners

# TODO/THINK: should RNN encoder return multiple outputs in case it contains multiple recurrent layers?
#      --> can not train multi-output blocks directly
#   -> what about RNN decoder? multiple inputs?



# TODO: ENABLE APPROPRIATE SUPPORT FOR MULTI-OUTPUT BLOCKS ::!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



# TODO: test RNN decoder
# + TODO: implement neg log prob loss for sequences (automatic detection or explicit setting?)