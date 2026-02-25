from data_serializer import DataSerializer

def main():
    file = "./dataset/bank_10.pkl"
    data = DataSerializer.deserialize(file)
    print(data)


if __name__ == '__main__':
    main()