# shaper

![](./data/flower/flower-1000-all.jpg)

## Usage

```
python main.py --n N --input INPUT --output OUTPUT render-mode RENDER_MODE --alpha ALPHA
```

| Arg           | Default   | Description                   |
|---------------|-----------|-------------------------------|
| n             | n/a       | number of shapes to draw      |
| input         | n/a       | path to the input image       |
| output        | n/a       | path to the output image, if not present, the result won't be saved      |
| render-mode   | 2         | 0 - waits for button press after every shape, 1 - draws without waiting, 2 - no render |
| alpha         | 0.5       | alpha value of the shapes - between 0 and 1 |