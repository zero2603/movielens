## Mô tả
- Tập dữ liệu MovieLens 1M<br /> 
- Project được xây dựng dựa trên framework Django<br />
- Code có tham khảo từ https://github.com/khanhnamle1994/movielens và machinelearningcoban.com

## Cài đặt
- Cài đặt Django ```pip install django```
- Chạy lệnh ```python manage.py migrate``` để tạo bảng session (dùng lưu các biến session trong file views.py)
- Cài đặt numpy, pandas,...
- Mở terminal trong thư mục project, chạy lệnh ``` python manage.py runserver ```
- Truy cập đường dẫn: http://127.0.0.1:8000/
- Lưu ý: login với username là 1 hoặc 2 hoặc 3, password tùy ý (đã build sẵn recommend cho user có id thuộc [1,2,3])
