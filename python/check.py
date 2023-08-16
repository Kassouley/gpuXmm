from math import log10, floor
import sys


def main() :
    try:
        f1 = open(f'{sys.argv[1]}')
        f2 = open(f'{sys.argv[2]}')
        values = []
        for line in f1:
            values += line.split()
        values_f1 = [float(value) for value in values]
        values = []
        for line in f2:
            values += line.split()
        values_f2 = [float(value) for value in values]

        errors = [abs(element1 - element2) for (element1, element2) in zip(values_f1, values_f2)]
        max_error=max(errors)
        if max_error == 0 :
            print("Kernel output is \u001b[42mcorrect\033[0m.")
        else :
            exponent = floor(log10(abs(max_error)))
            if exponent < -2 :
                print(f"Kernel output is \u001b[42mcorrect\033[0m with a max error of 10^{exponent}")
            else :
                print(f"Kernel output is \u001b[41mincorrect\033[0m (error max of 10^{exponent})")
    except IOError:
        print("Cannot open file")
        sys.exit(-1)

if __name__ == '__main__':
    main()