FasdUAS 1.101.10   ��   ��    k             j     �� �� 0 
num_epochs    m     ����     	  j    �� 
�� 0 
batch_size   
 m    ���� 
 	     j    �� �� 0 learning_rate    m       ?�������      j   	 �� �� 0 momentum    m   	 
   ?�������      l     ��������  ��  ��        l   � ����  O    �    O   �    k   �       r         4    �� !
�� 
NmTb ! m     " " � # #  N E T   o      ���� 0 net     $ % $ r    , & ' & K    * ( ( �� ) *�� 0 weights   ) 4    �� +
�� 
NmTb + m     , , � - -  F C 1 * �� . /�� 0 grad   . 4    �� 0
�� 
NmTb 0 m     1 1 � 2 2  F C 1 _ G R A D / �� 3 4�� 0 grad_acc   3 4    #�� 5
�� 
NmTb 5 m   ! " 6 6 � 7 7  F C 1 _ G R A D _ A C C 4 �� 8���� 0 step   8 4   $ (�� 9
�� 
NmTb 9 m   & ' : : � ; ;  F C 1 _ S T E P��   ' o      ���� 0 fc1   %  < = < r   - O > ? > K   - K @ @ �� A B�� 0 weights   A 4   . 4�� C
�� 
NmTb C m   0 3 D D � E E  F C 2 B �� F G�� 0 grad   F 4   5 ;�� H
�� 
NmTb H m   7 : I I � J J  F C 2 _ G R A D G �� K L�� 0 grad_acc   K 4   < B�� M
�� 
NmTb M m   > A N N � O O  F C 2 _ G R A D _ A C C L �� P���� 0 step   P 4   C I�� Q
�� 
NmTb Q m   E H R R � S S  F C 2 _ S T E P��   ? o      ���� 0 fc2   =  T U T r   P r V W V K   P n X X �� Y Z�� 0 weights   Y 4   Q W�� [
�� 
NmTb [ m   S V \ \ � ] ]  F C 3 Z �� ^ _�� 0 grad   ^ 4   X ^�� `
�� 
NmTb ` m   Z ] a a � b b  F C 3 _ G R A D _ �� c d�� 0 grad_acc   c 4   _ e�� e
�� 
NmTb e m   a d f f � g g  F C 3 _ G R A D _ A C C d �� h���� 0 step   h 4   f l�� i
�� 
NmTb i m   h k j j � k k  F C 3 _ S T E P��   W o      ���� 0 fc3   U  l m l r   s � n o n J   s | p p  q r q o   s t���� 0 fc1   r  s t s o   t w���� 0 fc2   t  u�� u o   w z���� 0 fc3  ��   o o      ���� 
0 layers   m  v w v r   � � x y x n   � � z { z 1   � ���
�� 
time { l  � � |���� | I  � �������
�� .misccurdldt    ��� null��  ��  ��  ��   y o      ���� 0 
start_time   w  } ~ } l  � ���������  ��  ��   ~   �  l  � ��� � ���   � - ' initialize weights and zero step value    � � � � N   i n i t i a l i z e   w e i g h t s   a n d   z e r o   s t e p   v a l u e �  � � � X   � ��� � � k   � � �  � � � X   � � ��� � � r   � � � � � \   � � � � � ]   � � � � � l  � � ����� � I  � �������
�� .sysorandnmbr    ��� nmbr��  ��  ��  ��   � m   � �����  � m   � �����  � n       � � � 1   � ���
�� 
NMCv � o   � ����� 0 c  �� 0 c   � n  � � � � � 2   � ���
�� 
NmCl � n   � � � � � 1   � ���
�� 
NMTc � n   � � � � � o   � ����� 0 weights   � o   � ����� 	0 layer   �  � � � l  � ���������  ��  ��   �  ��� � X   � ��� � � r   � � � � � m   � �����   � n       � � � 1   � ���
�� 
NMCv � o   � ����� 0 c  �� 0 c   � n  � � � � � 2   � ���
�� 
NmCl � n   � � � � � 1   � ���
�� 
NMTc � n   � � � � � o   � ����� 0 step   � o   � ����� 	0 layer  ��  �� 	0 layer   � o   � ����� 
0 layers   �  � � � l ��������  ��  ��   �  � � � l �� � ���   �   train    � � � �    t r a i n �  � � � r   � � � 4  �� �
�� 
NmTb � m  
 � � � � �  i r i s _ t r a i n � o      ���� 	0 train   �  � � � r  " � � � I �� ���
�� .corecnte****       **** � n  � � � 2 ��
�� 
NMRw � o  ���� 	0 train  ��   � o      ���� 0 num_samples   �  � � � r  #0 � � � ^  #, � � � o  #&���� 0 num_samples   � o  &+���� 0 
batch_size   � o      ���� 0 num_batches   �  ��� � Y  1� ��� � ��� � k  ?� � �  � � � r  ?D � � � m  ?@����   � o      ���� 0 running_loss   �  � � � l EE��������  ��  ��   �  � � � Y  E5 ��� � ��� � k  Q0 � �  � � � l QQ�� � ���   �   zero gradients    � � � �    z e r o   g r a d i e n t s �  � � � X  Q� ��� � � X  g� ��� � � r  �� � � � m  ������   � n       � � � 1  ����
�� 
NMCv � o  ������ 0 c  �� 0 c   � n ju � � � 2  qu��
�� 
NmCl � n  jq � � � 1  mq��
�� 
NMTc � n  jm � � � o  km���� 0 grad_acc   � o  jk���� 	0 layer  �� 	0 layer   � o  TW���� 
0 layers   �  � � � l ����������  ��  ��   �  � � � l ���� � ���   �   compute gradients    � � � � $   c o m p u t e   g r a d i e n t s �  � � � Y  �� ��� � �� � k  �� � �  � � � l ���~ � ��~   �   load the sample    � � � �     l o a d   t h e   s a m p l e �  � � � l ���} �}    ? 9 everything is 1-based, except our table witch is 2-based    � r   e v e r y t h i n g   i s   1 - b a s e d ,   e x c e p t   o u r   t a b l e   w i t c h   i s   2 - b a s e d �  r  �� [  �� [  ��	
	 ]  �� l ���|�{ \  �� o  ���z�z 	0 batch   m  ���y�y �|  �{   o  ���x�x 0 
batch_size  
 o  ���w�w 0 batch_sample   m  ���v�v  o      �u�u 0 
sample_num    r  �� n  �� 4  ���t
�t 
NMRw o  ���s�s 0 
sample_num   o  ���r�r 	0 train   o      �q�q 
0 sample    r  �� n  �� 4  ���p
�p 
NMRw m  ���o�o  o  ���n�n 0 net   o      �m�m 	0 input    Y  � �l!"�k  r  ��#$# n  ��%&% 1  ���j
�j 
NMCv& n  ��'(' 4  ���i)
�i 
NmCl) o  ���h�h 0 i  ( o  ���g�g 
0 sample  $ n      *+* 1  ���f
�f 
NMCv+ n  ��,-, 4  ���e.
�e 
NmCl. o  ���d�d 0 i  - o  ���c�c 	0 input  �l 0 i  ! m  ���b�b " n  ��/0/ l ��1�a�`1 I ���_2�^
�_ .corecnte****       ****2 2 ���]
�] 
NmCl�^  �a  �`  0 o  ���\�\ 	0 input  �k   343 l �[�Z�Y�[  �Z  �Y  4 565 l �X78�X  7   update the gradients   8 �99 *   u p d a t e   t h e   g r a d i e n t s6 :;: X  m<�W=< Y  h>�V?@�U> k  .cAA BCB r  .:DED n  .6FGF 4  16�TH
�T 
NmClH o  45�S�S 0 i  G n  .1IJI o  /1�R�R 0 grad_acc  J o  ./�Q�Q 	0 layer  E o      �P�P 0 cur  C KLK r  ;GMNM n  ;COPO 4  >C�OQ
�O 
NmClQ o  AB�N�N 0 i  P n  ;>RSR o  <>�M�M 0 grad  S o  ;<�L�L 	0 layer  N o      �K�K 	0 delta  L T�JT r  HcUVU [  H[WXW l HOY�I�HY n  HOZ[Z 1  KO�G
�G 
NMCv[ o  HK�F�F 0 cur  �I  �H  X ^  OZ\]\ l OV^�E�D^ n  OV_`_ 1  RV�C
�C 
NMCv` o  OR�B�B 	0 delta  �E  �D  ] o  VY�A�A 0 num_batches  V n      aba 1  ^b�@
�@ 
NMCvb o  [^�?�? 0 cur  �J  �V 0 i  ? m  �>�> @ n  )cdc l !)e�=�<e I !)�;f�:
�; .corecnte****       ****f 2 !%�9
�9 
NmCl�:  �=  �<  d n  !ghg o  !�8�8 0 grad  h o  �7�7 	0 layer  �U  �W 	0 layer  = o  
�6�6 
0 layers  ; iji l nn�5�4�3�5  �4  �3  j k�2k r  n�lml [  n�non o  nq�1�1 0 running_loss  o l q�p�0�/p n  q�qrq 1  ~��.
�. 
NMCvr n  q~sts 4  y~�-u
�- 
NmClu m  |}�,�, t n  qyvwv 4  ry�+x
�+ 
NmCRx m  uxyy �zz  A 9w o  qr�*�* 0 net  �0  �/  m o      �)�) 0 running_loss  �2  �� 0 batch_sample   � m  ���(�(  � o  ���'�' 0 
batch_size  �   � {|{ l ���&�%�$�&  �%  �$  | }~} l ���#��#     step   � ��� 
   s t e p~ ��"� X  �0��!�� Y  �+�� ���� k  �&�� ��� l ������  �   update step value   � ��� $   u p d a t e   s t e p   v a l u e� ��� r  ����� n  ����� 4  ����
� 
NmCl� o  ���� 0 i  � n  ����� o  ���� 0 step  � o  ���� 	0 layer  � o      �� 0 cur  � ��� r  ����� n  ����� 4  ����
� 
NmCl� o  ���� 0 i  � n  ����� o  ���� 0 grad_acc  � o  ���� 	0 layer  � o      �� 	0 delta  � ��� r  ����� \  ����� ]  ����� o  ���� 0 momentum  � l ������ n  ����� 1  ���
� 
NMCv� o  ���� 0 cur  �  �  � ]  ����� o  ���� 0 learning_rate  � l ������ n  ����� 1  ���
� 
NMCv� o  ���
�
 	0 delta  �  �  � n      ��� 1  ���	
�	 
NMCv� o  ���� 0 cur  � ��� l ������  �  �  � ��� l ������  �   update weight value   � ��� (   u p d a t e   w e i g h t   v a l u e� ��� r  ���� n  ����� 4  ����
� 
NmCl� o  ���� 0 i  � n  ����� o  ���� 0 weights  � o  ��� �  	0 layer  � o      ���� 0 cur  � ��� r  ��� n  
��� 4  
���
�� 
NmCl� o  	���� 0 i  � n  ��� o  ���� 0 step  � o  ���� 	0 layer  � o      ���� 	0 delta  � ���� r  &��� [  ��� l ������ n  ��� 1  ��
�� 
NMCv� o  ���� 0 cur  ��  ��  � l ������ n  ��� 1  ��
�� 
NMCv� o  ���� 	0 delta  ��  ��  � n      ��� 1  !%��
�� 
NMCv� o  !���� 0 cur  ��  �  0 i  � m  ������ � n  ����� l �������� I �������
�� .corecnte****       ****� 2 ����
�� 
NmCl��  ��  ��  � n  ����� o  ������ 0 step  � o  ������ 	0 layer  �  �! 	0 layer  � o  ������ 
0 layers  �"  �� 	0 batch   � m  HI����  � o  IL���� 0 num_batches  ��   � ��� l 66��������  ��  ��  � ��� l 66������  � 
  Log   � ���    L o g� ��� I 6E�����
�� .NMTbARafnull���     NmCR� n  6A��� 4 <A���
�� 
NMRw� m  ?@������� 4  6<���
�� 
NmTb� m  8;�� ���  L O G��  � ���� O  F���� k  T��� ��� r  T`��� o  TU���� 	0 epoch  � n      ��� 1  [_��
�� 
NMCv� 4  U[���
�� 
NmCl� m  YZ���� � ��� r  as��� ^  ah��� o  ad���� 0 running_loss  � o  dg���� 0 num_samples  � n      � � 1  nr��
�� 
NMCv  4  hn��
�� 
NmCl m  lm���� �  r  t� l ty���� I ty������
�� .misccurdldt    ��� null��  ��  ��  ��   n       1  ���
�� 
NMCv 4  y��	
�� 
NmCl	 m  }~����  
��
 r  �� \  �� l ������ n  �� 1  ����
�� 
time l ������ I ��������
�� .misccurdldt    ��� null��  ��  ��  ��  ��  ��   o  ������ 0 
start_time   n       1  ����
�� 
NMCv 4  ����
�� 
NmCl m  ������ ��  � n  FQ 4 LQ��
�� 
NMRw m  OP������ 4  FL��
�� 
NmTb m  HK �  L O G��  �� 	0 epoch   � m  45����  � o  5:���� 0 
num_epochs  ��  ��    1    
��
�� 
NmAS  n      4    ��
�� 
docu m    ����  m     �                                                                                  NMBR  alis    &  Macintosh HD                   BD ����Numbers.app                                                    ����            ����  
 cu             Applications  /:Applications:Numbers.app/     N u m b e r s . a p p    M a c i n t o s h   H D  Applications/Numbers.app  / ��  ��  ��     !  l     ��������  ��  ��  ! "��" l     ��������  ��  ��  ��       ��#����  $��  # ������������ 0 
num_epochs  �� 0 
batch_size  �� 0 learning_rate  �� 0 momentum  
�� .aevtoappnull  �   � ****�� �� 
$ ��%����&'��
�� .aevtoappnull  �   � ****% k    �((  ����  ��  ��  & �������������� 	0 layer  �� 0 c  �� 	0 epoch  �� 	0 batch  �� 0 batch_sample  �� 0 i  ' 6������ "���� ,�� 1�� 6�� :���� D I N R�� \ a f j������������������������ ���������~�}�|�{�z�y�xy��w�v
�� 
docu
�� 
NmAS
�� 
NmTb�� 0 net  �� 0 weights  �� 0 grad  �� 0 grad_acc  �� 0 step  �� �� 0 fc1  �� 0 fc2  �� 0 fc3  �� 
0 layers  
�� .misccurdldt    ��� null
�� 
time�� 0 
start_time  
�� 
kocl
�� 
cobj
�� .corecnte****       ****
�� 
NMTc
�� 
NmCl
�� .sysorandnmbr    ��� nmbr
�� 
NMCv�� 	0 train  
�� 
NMRw�� 0 num_samples  � 0 num_batches  �~ 0 running_loss  �} 0 
sample_num  �| 
0 sample  �{ 	0 input  �z 0 cur  �y 	0 delta  
�x 
NmCR
�w .NMTbARafnull���     NmCR�v �����k/�*�,�*��/E�O�*��/�*��/�*��/�*��/�E�O�*�a /�*�a /�*�a /�*�a /�E` O�*�a /�*�a /�*�a /�*�a /�E` O�_ _ mvE` O*j a ,E` O w_ [a a l  kh   1��,a !,a "-[a a l  kh *j #l k�a $,F[OY��O )��,a !,a "-[a a l  kh j�a $,F[OY��[OY��O*�a %/E` &O_ &a '-j  E` (O_ (b  !E` )Oskb   kh jE` *O�k_ )kh  D_ [a a l  kh   )��,a !,a "-[a a l  kh j�a $,F[OY��[OY��O �kb  kh �kb   �kE` +O_ &a '_ +/E` ,O�a 'k/E` -O 1k_ -a "-j  kh _ ,a "�/a $,_ -a "�/a $,F[OY��O h_ [a a l  kh   Mk��,a "-j  kh ��,a "�/E` .O��,a "�/E` /O_ .a $,_ /a $,_ )!_ .a $,F[OY��[OY��O_ *�a 0a 1/a "k/a $,E` *[OY�O �_ [a a l  kh   �k��,a "-j  kh ��,a "�/E` .O��,a "�/E` /Ob  _ .a $, b  _ /a $, _ .a $,FO��,a "�/E` .O��,a "�/E` /O_ .a $,_ /a $,_ .a $,F[OY��[OY�r[OY�O*�a 2/a 'i/j 3O*�a 4/a 'i/ M�*a "k/a $,FO_ *_ (!*a "l/a $,FO*j *a "m/a $,FO*j a ,_ *a "a 5/a $,FU[OY��UUascr  ��ޭ