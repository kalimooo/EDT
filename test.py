import torch

def main():
    print("Program starting... Is cuda available?")
    print(torch.cuda.is_available())

if __name__ == "__main__":
    main()