def gen_train(train_df):
    # ['trajectory', 'user_index', 'day']
    records = []
    for index, row in train_df.iterrows():
        seq, user_index, day = row['trajectory'], row['user_index'], row['year'] + row['month'] + row['day']
        records.append([seq, user_index, day])
    print("All train length is " + str(len(records)))
    return records

def gen_test(self):
    # ['trajectory', 'masked_pos', 'masked_tokens']
    test_df = self.test_df
    records = []
    for index, row in test_df.iterrows():
        seq, masked_pos, masked_tokens = row['trajectory'], row['masked_pos'], row['masked_tokens']
        user_index, day = row['user_index'], row['day']
        seq, masked_pos, masked_tokens = list(seq.split()), list(map(int, masked_pos.split())), \
                                         list(map(int, masked_tokens.split()))
        records.append([seq, masked_pos, masked_tokens, user_index, day])
    print("All test length is " + str(len(records)))
    return records