import os

# os.system('rm -f results/*52260409*pt*')

use_id = ['52260390', '52260407', '52260429', '52229020', '52260377', '52260275', '52903345', '52428066', '52167752', '52455797', '52260390', '52376453', '52182199', '53113236', '51199946', '52277589', '52378572', '52330995', '52260407', '53228692', '52455797', '52428866', '52455815', '52332090', '52332131', '52280541', '52866854', '52866853', '52866853', '52866856', '52314780', '52455815', '52428199', '52903353', '52351432', '53246902', '53246898', '53246913', '53246882', '53246832', '52420528', '52260390', '53246979', '53246952', '53246932', '52734675', '52734677', '52734680', '52734681', '52734682', '52734683', '52734685', '52734686', '52734687', '52806123', '52806124', '52806125', '52806127', '52865694', '52866132', '52866270', '52866443', '52866444', '52866445', '52866446', '52866447', '52866448', '52866449', '52866450', '52866563', '52866589', '52866590', '52866655', '52866656', '52866657', '52866658', '52866660', '52866664', '52866665', '52866667', '52866670', '53113237', '53113236', '53113234', '53113233', '53113232', '53112988', '53112970', '53599310', '53599423', '53599430', '53599556', '53599775', '53599872', '53599886', '53599898', '53599900', '53634552', '53634553', '53645306', '53645309', '53645311', '53645314', '53645316', '53645396', '53657706', '53657712', '53657713', '53657715', '53660105', '53660106', '53673341', '53673343', '53689162', '53716662', '53725594', '53725598', '53725599', '53725671', '53726271', '53726277', '53729466', '53729468', '53730909', '53730917', '53738139', '53738140', '53822038', '53822887', '53944286', '53944314', '53967350', '53967352', '53967357', '53967362', '53967366', '53967567', '53967575', '53967582', '53967590', '53967592', '53968070',]

file_names = os.listdir('./results/')

# print(file_names)
# print(type(file_names))

for file_name in file_names:
    if 'samplewise' in file_name and 'pt' in file_name and '20220516' in file_name:
        delete_flag = True
        for job_id in use_id:
            if job_id in file_name:
                delete_flag = False
                break
        if delete_flag:
            # os.system('rm -f ./results/{}'.format(file_name))
            print(file_name)
