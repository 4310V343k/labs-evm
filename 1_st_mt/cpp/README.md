# Переписывание Rust проекта на C++

Этот каталог содержит переписанную на C++ версию исходных Rust‑крэйтов:
- `common`
- `integral_single`
- `integral_parallel`
- `integral_opencl` (опционально, через OpenCL C API)
- `integral_parallel_omp` (опционально, реализация на OpenMP)
- `app` (аналог корневого бинарника с таблицей результатов)

## Зависимости
- Компилятор с поддержкой C++17 (MSVC / clang / gcc)
- CMake >= 3.16
- (Опционально) OpenCL SDK / драйвер (например, Intel, NVIDIA, AMD)
- (Опционально) Поддержка OpenMP в компиляторе (MSVC / clang / gcc)

## Сборка (Windows PowerShell)
```powershell
cd cpp_rewrite
cmake -S . -B build -DENABLE_OPENCL=ON -DENABLE_OPENMP=ON
cmake --build build --config Release
```
Исполняемый файл: `build/app/weier_benchmark(.exe)`

## Запуск
```powershell
./build/app/Release/weier_benchmark.exe
```
(или без `Release/` если генератор создал сразу в корне целевой папки)

## Отличия от Rust версии
- Табличный вывод реализован вручную (ширины столбцов фиксированные)
- Проверка результата оставлена с теми же допусками
- OpenCL реализация использует чистый C API вместо high-level обёртки
- Обработка ошибок OpenCL — через `last_error()`
- Добавлена отдельная версия параллельного интегрирования на OpenMP (сравнение трёх CPU реализаций: single, async-пулы, OpenMP)

## TODO / Возможные улучшения
- Добавить detection CPU fallback если нет GPU (использовать CL_DEVICE_TYPE_ALL)
- Параметры (n, steps) вывести в аргументы командной строки
- Unit-тесты (GoogleTest / Catch2)
- Настройки числа потоков OpenMP через переменную окружения OMP_NUM_THREADS либо параметр

