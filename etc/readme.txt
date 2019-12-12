<������ �� Ȱ�� �����͸� Ȱ���Ͽ� ���� ��ġ�� ����� ��Ż ����>
								-���ڼ���-
1. �ڵ� ���࿡ �ʿ��� ��Ű�� �� ���̺귯��
    - Anaconda3�� ���Ե� ��Ű��
    - pandas
    - numpy
    - sklearn
    - pickle ��
    - lightgbm (version 2.2.4)
        version 2.2.3������ custom objective function�� ���Ե� lightgbm ���� �����ϴ� �κп��� �����߻�
        �������� �ڵ带 ������ ���ؼ��� �ݵ�� 2.2.4 ������ �ʿ���

2. ����ȯ��
    - Anaconda3 ����� python3 ȯ��

3. �ڵ� ���� ���� �� ���
    1. ���ڼ���/preprocess/preprocess.py
        - ���ڼ���/raw ������ �ִ� Train �����Ϳ� Test �����͸� �ҷ��ͼ� ��ó�� �ϴ� �ڵ�
        - ���ڼ���/preprocess ������ �� �н��� ���� CSV ���� ����
            * train_preprocess_1.csv ���� (Survival Time�� �����ϱ� ���� ���� �н��ϱ� ���� Train ������)
            * train_preprocess_2.csv ���� (Amount Spent�� �����ϱ� ���� ���� �н��ϱ� ���� Train ������
        - ���ڼ���/preprocess ������ Test �����͸� �����ϱ� ���� ���� input ������(CSV ����) ����
            * test1_preprocess_1.csv (Test1�� Survival Time�� �����ϱ� ���� Test1 ��ó�� ������)
	* test1_preprocess_2.csv (Test1�� Amount Spent�� �����ϱ� ���� Test1 ��ó�� ������)
	* test2_preprocess_1.csv (Test2�� Survival Time�� �����ϱ� ���� Test2 ��ó�� ������)
	* test2_preprocess_2.csv (Test2�� Amount Spent�� �����ϱ� ���� Test2 ��ó�� ������)
    2.���ڼ���/model/create_model.py
        - ���� �����ϰ�, ���ڼ���/preprocess ������ �ִ� ��ó�� �����͸� �ҷ��ͼ� �н��� ������ �� �н��� �� ��ü�� ����
            * train_preprocess_1.csv ���Ϸ� Survival Time ������ ���� �𵨵��� �н�
            * train_preprocess_2.csv ���Ϸ� Amount Spent ������ ���� ���� �н�
        - ���ڼ���/model ������ Test ������ ������ ���� �� ��ü ����
            * final_model_1.sav (Survival Time - ����/������ ������ �����ϱ� ���� �з� �ӻ�� ��)
            * final_model_2.sav (Survival Time - ������ ������ Survival Time�� �����ϱ� ���� ȸ�� �ӻ�� ��)
            * final_model_3.sav (Amount Spent - ������ Amount spent�� �����ϱ� ���� ȸ�� ��)
    3. ���ڼ���/predict/predict.py ����
        - ���ڼ���/model ������ ������ �� ��ü�� �ҷ��ͼ� Test �����͸� ����
        - ���� ������ Test �������� Survival Time�� Amount Spent�� ���� ����� ����
            * Test1 : test1_predict.csv
            * Test2 : test2_predict.csv

4. �ڵ� ����
    - readme.md ���Ͽ� �ڼ��� ����