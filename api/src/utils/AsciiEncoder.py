import string
import sklearn.preprocessing as skl_preproc


class AsciiEncoder:
    AVAILABLE_CHARS = ' ' + string.ascii_lowercase

    @staticmethod
    def convert_characters_to_indexes(characters_array):
        encoder = skl_preproc.LabelEncoder()
        encoder.fit(list(AsciiEncoder.AVAILABLE_CHARS))

        return encoder.transform(list(characters_array))

    @staticmethod
    def convert_indexes_to_characters(indexes_array):
        encoder = skl_preproc.LabelEncoder()
        encoder.fit(list(AsciiEncoder.AVAILABLE_CHARS))

        return encoder.inverse_transform(indexes_array)


if __name__ == '__main__':
    print(AsciiEncoder.AVAILABLE_CHARS)
    print(AsciiEncoder.convert_characters_to_indexes(AsciiEncoder.AVAILABLE_CHARS))