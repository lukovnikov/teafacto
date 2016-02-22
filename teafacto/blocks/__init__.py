# TODO: add ContrastModel and support in the trainer. This is for negative sampling models
#           - the first dimension of data separates positive [0] from all negative examples [1:]
#           - objective function operates between positive and negative samples
# TODO: implement example dynamic data feed
# TODO: implement more objectives
# TODO: add (auto)saving --> in trainer
# TODO: how to chain models? --> model should be a block
# TODO: converge blocks and models --> named inputs