# AVP_LAB_7
MPI  +  FISH EYE  +  BILINEAR Interpolation

"AVP5.pdf" Вариант 2

Маркер - красная(C =255 0 0) окружность, центр которого совпадает с центром изображения. Преобразование - рыбий глаз. После обратного преобразования
должна получиться, окружность радиус окружность должен быть равен 10% от меньшей стороны
изображения. Метод интерполяции - BELINEAR.

Для 7 лабы необходимо сделать такую же обработку изображения, но на двух компьютерах с использованием MPI.

принцип работы прост:
1) 2 компа (ПК1, ПК2) открывают 1 картинку
2) каждый ПК обрабатывает по половинке изображения
3) ПК2 отправляет половину картинки ПК1
4) ПК1 склеивает всё и выводит результат

_____доп. после выполнения каждого преобразования выводиться результат на двух ПК

В чем прикол преобразования хафа для этой лабы я так и не понял, потому что основной прикол тут это рыбий глаз и интерполяция.

В теории должно обрабатывать картинки любого разрешения, но выше чем 700х1000 у меня не получилось, после рыбьего глаза появляется просто белая картинка, 
возможно связана с тем что числа, после вычислений, получаются больше чем может позволить float. Если у вас получиться зарендерить картинку 1920х1080 то вы красава

Для работы данной программы, необходимо два ПК, на одном стоит win_server, на втором просто win, чтобы можно было поставить opencv, cuda и mpi.
Win_server нужен для того чтобы создать доменную зону, а потом второй пк ввести в этот домен. Обязательно отключите брэндмауэр.
После настройки домена, войдите в одну учётную запись с двух ПК. Между ПК создайте общую сетевую папку, закиньте туда проект и работайте только в этой папке

___P.S. 1)Этот код у того, кто преподаёт данный предмет тоже есть 2) я использовал ms_mpi 3) чтобы юзать mpi по сети вам нужен smpd 


#УДАЧИ
