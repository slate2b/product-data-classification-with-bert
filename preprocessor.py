from datasets import load_dataset


def prepare_dataset(source_fpath):
    """

    :param source_fpath:
    :return:
    """

    file_dict = {
        "main": source_fpath,
    }

    original = load_dataset(
        'csv',
        data_files=file_dict,
        delimiter=',',
        column_names=['text', 'label'],
        skiprows=1
    )

    #####
    # use train_test_split to create train, test, and validation splits
    #

    # shuffle the original dataset
    original['main'] = original['main'].shuffle(seed=1)

    # split the shuffled dataset into a 60/40 train/test
    dataset = original['main'].train_test_split(train_size=0.7)

    # create a new temporary dataset to split the test split into 2 splits
    # this will result in a train and test split here, too
    temporary_test_validation = dataset['test'].train_test_split(train_size=0.5)

    # pop the test split from the main dataset since we will be using the splits below
    dataset.pop('test')

    # create a validation split based on the temporary_test_validation 'train' split
    # this will result in a train, test, and validation split (copy of train)
    temporary_test_validation['validation'] = temporary_test_validation['train']

    # pop the 'train' split from the temporary dataset
    temporary_test_validation.pop('train')

    # update the main dataset to include the new test and validation splits
    dataset.update(temporary_test_validation)

    return dataset
