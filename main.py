import torchvision


dataset= torchvision.datasets.Flowers102(
                            root='data/',
                            download=False,
                            )

data = dataset[0]

print(type(data[0]))
print(type(data[1]))

print(data[0].size)

                    