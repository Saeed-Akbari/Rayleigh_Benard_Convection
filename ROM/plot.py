import numpy as np
import matplotlib.pyplot as plt
import yaml


def main():

    with open('input.yaml') as file:    
        input_data = yaml.load(file, Loader=yaml.FullLoader)
    file.close()

if __name__ == "__main__":
    main()
