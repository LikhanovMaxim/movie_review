def run():
    stars = ['1', '7', '10']
    for i in range(len(stars)):
        if int(stars[i]) > 5:
            stars[i] = 1
        else:
            stars[i] = 0
    for star in stars:
        print(star)


if __name__ == '__main__':
    run()
