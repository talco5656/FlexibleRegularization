import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from torch_code.torch_tests import TorchExample, parse_args, DataTuple
import numpy as np


class KNNTest(TorchExample):
    def __init__(self, args):
        TorchExample.__init__(self, args=args)

    def get_feature_extractor(self, model):
        # model_ft = models.resnet18(pretrained=True)
        ### strip the last layer
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        return feature_extractor
        ### check this works
        # x = torch.randn([1,3,224,224])
        # output = feature_extractor(x) # output now has the features corresponding to input x
        # print(output.shape)

    def knn_prediction(self, model: torch.nn.Sequential, datatuple: DataTuple):
        feature_extractor = self.get_feature_extractor(model)
        train_knn_x, train_knn_y = self.get_features(feature_extractor, datatuple.train)
        val_knn_x, val_knn_y = self.get_features(feature_extractor, datatuple.val)
        neigh = KNeighborsClassifier(n_neighbors=10)
        train_knn_x = np.reshape(train_knn_x, (train_knn_x.shape[0], -1))
        neigh.fit(train_knn_x, train_knn_y)
        val_knn_x = np.reshape(val_knn_x, (val_knn_x.shape[0], -1))
        prediction = neigh.predict(val_knn_x)
        accuracy = accuracy_score(prediction, val_knn_y)
        return accuracy
        # print(neigh.predict([[1.1]]))
        # print(neigh.predict_proba([[0.9]]))

    def get_features(self, model, dataloader):
        features = []
        labels = []
        for x, y in dataloader:
            features.append(model(x).detach().numpy())
            labels.append(y.detach().numpy())
        features = np.concatenate(features)
        labels = np.concatenate(labels)
        return features, labels

    def create_dataloaders_subsets(self):
        first_subset_dict = {}
        second_subset_dict = {}
        first_cifar_subset = self.get_pytorch_cifar_data()[
            0]._asdict()  # CIFARSubset(subsets["first split data"], subsets["first split labels"])
        second_cifar_subset = self.get_pytorch_cifar_data()[
            0]._asdict()  # CIFARSubset(subsets["second split data"], subsets["second split labels"])
        for key, subset in self.data._asdict().items():
            subsets = self.divide_to_sub_sets(subset.dataset)

            first_cifar_subset[key].data = subsets["first split data"]
            first_cifar_subset[key].targets = subsets["first split labels"]
            second_cifar_subset[key].data = subsets["second split data"]
            second_cifar_subset[key].targets = subsets["second split labels"]
            # first_subset_dict[key] = first_cifar_subset
            # second_subset_dict[key] = second_cifar_subset
        first_datatuple = DataTuple(**first_cifar_subset)
        second_datatuple = DataTuple(**second_cifar_subset)
        return first_datatuple, second_datatuple



    def knn_experiment(self):
        first_datatuple, second_datatuple = self.create_dataloaders_subsets()

        self.data = first_datatuple  # self.get_data_loaders(datatuple=first_datatuple)
        # train
        result_df, original_model, adaptive_model = self.train_and_eval()
        # knn results

        original_accuracy = self.knn_prediction(original_model, second_datatuple)
        adaptive_accuracy = self.knn_prediction(adaptive_model, second_datatuple)
        print(f"original accuracy: {original_accuracy},"
              f"adaptive accuracy: {adaptive_accuracy}")


def test1():
    args = parse_args()
    # mean_and_ci_result(args)
    torch_example = KNNTest(args)
    torch_example.knn_experiment()
    # d = torch_example.get_cifar10_data()
    # torch_example.divide_to_sub_sets(d)
    # torch_example.mean_and_ci_result()


if __name__ == "__main__":
    test1()
    # main()
