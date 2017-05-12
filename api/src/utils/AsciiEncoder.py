import string
import sklearn.preprocessing as skl_preproc


class AsciiEncoder:
    AVAILABLE_CHARS = ' ' + string.digits + string.ascii_lowercase

    @staticmethod
    def convert_characters_to_indexes(characters_array):
        encoder = skl_preproc.LabelEncoder()
        encoder.fit(list(AsciiEncoder.AVAILABLE_CHARS))

        return encoder.transform(characters_array)

    @staticmethod
    def convert_indexes_to_characters(indexes_array):
        encoder = skl_preproc.LabelEncoder()
        encoder.fit(list(AsciiEncoder.AVAILABLE_CHARS))

        return encoder.inverse_transform(indexes_array)

