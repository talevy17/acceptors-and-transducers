import random

import rstr as rstr


def generate_word(reg_exp):
    word = rstr.xeger(reg_exp)
    return word

def generate_words(num_of_words,file_name,regex):
    temp_words =[]
    with open('./Data/{0}'.format(file_name), mode='w') as file:
        for i in range(num_of_words):
            word = generate_word(regex)
            #check if the word wasnt before
            if word in temp_words:
                i = i-1
                continue
            temp_words.append(word)
            file.write("{0}\n".format(word))
    file.close()

def generate_train_dev_test(num_of_words,file_name,regexes, isTest=False):
    temp_words =[]
    with open('./Data/{0}'.format(file_name), mode='w') as file:
        for i in range(num_of_words):
            label = random.randint(0,len(regexes)-1)
            regex = regexes[label]
            word = generate_word(regex)
            # check if the word wasnt before
            if word in temp_words:
                i = i - 1
                continue
            temp_words.append(word)
            if not isTest:
                file.write("{0}\t{1}\n".format(word,label))
            else:
                file.write("{0}\n".format(word))
        file.close()


if __name__ == "__main__":

    positive_regex =(r"[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+")
    negative_regex = (r"[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+")
    #generate_words(500,'pos_examples',positive_regex)
    #generate_words(500,'neg_examples',negative_regex)
    regexes = [negative_regex,positive_regex]
    generate_train_dev_test(500,'dev', regexes, isTest=False)
    generate_train_dev_test(500,'train', regexes, isTest=False)
    generate_train_dev_test(500,'test', regexes, isTest=True)


