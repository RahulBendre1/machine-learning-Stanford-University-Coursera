package com.mkis.assignments.gradientdescentandnormalequation;

        import org.apache.commons.math3.linear.*;
        import org.jfree.chart.ChartPanel;
        import org.jfree.chart.JFreeChart;
        import org.jfree.chart.axis.NumberAxis;
        import org.jfree.chart.axis.ValueAxis;
        import org.jfree.chart.plot.XYPlot;
        import org.jfree.chart.renderer.xy.XYItemRenderer;
        import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
        import org.jfree.data.xy.XYDataset;
        import org.jfree.data.xy.XYSeries;
        import org.jfree.data.xy.XYSeriesCollection;
        import org.jfree.ui.ApplicationFrame;
        import org.jfree.ui.RefineryUtilities;

        import java.io.BufferedReader;
        import java.io.FileReader;
        import java.text.DecimalFormat;
        import java.text.NumberFormat;
        import java.util.ArrayList;
        import java.util.List;
        import java.util.stream.DoubleStream;

//First weekly assignments part 1 Linear Regression with gradient descent vectorized solution

public class GradientDescentLinearRegressionOneVariableVectorized extends ApplicationFrame {

    private static String file = "D:\\Projects-repos\\MachineLearning\\src\\com\\mkis\\assignments\\gradientdescentandnormalequation\\data1.txt";
    private static double[][] data; // data array
    private static double[][] variablesWithOneArray;  //variable array with ones added to the first row (X matrix)
    private static double[] valuesArray;  // values array (y vector)
    private static int dataRow = 97; //number of rows in data set (m)
    private static int dataCols = 2; // number of columns in data set (n)
    private static double m; // number of training examples
    private static int n; // number of features + 1

    private static double alpha = 0.01; // learning rate
    private static int iterationsVar = 1;
    private static int iterations = 1500; // number of iterations for gradient descent

    private static RealMatrix X; // matrix X (Population of City in 10,000s)
    private static RealVector y; // vector y (Profit in $10,000s)
    private static RealVector theta; //vector theta (parameters)

    private static List<Double> xAxisValues = new ArrayList<>();
    private static List<Double> yAxisValues = new ArrayList<>();

    public GradientDescentLinearRegressionOneVariableVectorized(final String title) {

        super(title);

        // Create a single plot containing both the scatter and line
        XYPlot plot = new XYPlot();

        //SETUP SCATTER graph
        // Create the scatter data, renderer, and axis
        XYSeries series = new XYSeries("Profits depending on city population");
        for (int i = 0; i < xAxisValues.size(); i++) {
            series.add(xAxisValues.get(i), yAxisValues.get(i));
        }
        XYDataset dataSet = getData(series);
        XYItemRenderer rendererData = new XYLineAndShapeRenderer(false, true);   // Shapes only
        ValueAxis xAxisData = new NumberAxis("Population of City in 10,000s");
        ValueAxis yAxisData = new NumberAxis("Profit in $10,000s");
        xAxisData.setLowerBound(4);
        xAxisData.setUpperBound(25);
        yAxisData.setLowerBound(-5);
        yAxisData.setUpperBound(25);

        // Set the scatter data, renderer, and axis into plot
        plot.setDataset(0, dataSet);
        plot.setRenderer(0, rendererData);
        plot.setDomainAxis(0, xAxisData);
        plot.setRangeAxis(0, yAxisData);

        //SETUP LINE graph
        // Create the line data, renderer, and axis
        XYSeries lineSeries = new XYSeries("Regression function");
        for (int i = 0; i < xAxisValues.size(); i++) {
            lineSeries.add((double) i, -3.6302914394044015 + 1.1663623503355864 * i);
        }
        XYDataset lineDataSet = getLineData(lineSeries);
        XYItemRenderer rendererLine = new XYLineAndShapeRenderer(true, false);   // Lines only
        ValueAxis xAxisLine = new NumberAxis();
        ValueAxis yAxisLine = new NumberAxis();
        xAxisLine.setLowerBound(4);
        xAxisLine.setUpperBound(25);
        xAxisLine.setAxisLineVisible(false);
        yAxisLine.setLowerBound(-5);
        yAxisLine.setUpperBound(25);
        yAxisLine.setAxisLineVisible(false);

        // Set the line data, renderer, and axis into plot
        plot.setDataset(1, lineDataSet);
        plot.setRenderer(1, rendererLine);
        plot.setDomainAxis(1, xAxisLine);
        plot.setRangeAxis(1, yAxisLine);

        // Map the line to the second xAxis and second yAxis
        plot.mapDatasetToDomainAxis(1, 1);
        plot.mapDatasetToRangeAxis(1, 1);

        // Create the chart with the plot and a legend
        JFreeChart chart = new JFreeChart("Data set - Linear Regression", JFreeChart.DEFAULT_TITLE_FONT, plot, true);

        final ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(800, 800));
        setContentPane(chartPanel);
    }

    public static void main(String[] args) {

        loadData();
        createMatrixX();
        createVectorY();
        createTheta();
        createCostFunction();
        doGradientDescent();

        NumberFormat nf = new DecimalFormat("##.##");
        System.out.println("The value (profit) prediction in $s for city (population) size: 35,000:");
        System.out.println(nf.format((theta.getEntry(0) + theta.getEntry(1) * 3.5) * 10000));

        GradientDescentLinearRegressionOneVariableVectorized visualizationOfData =
                new GradientDescentLinearRegressionOneVariableVectorized("Visualization of data");
        visualizationOfData.pack();
        RefineryUtilities.centerFrameOnScreen(visualizationOfData);
        visualizationOfData.setResizable(false);
        visualizationOfData.setVisible(true);
    }

    //Get XYDataset data data for visualization
    private XYDataset getData(XYSeries series) {
        XYSeriesCollection xySeriesCollectionData = new XYSeriesCollection();
        xySeriesCollectionData.addSeries(series);
        return xySeriesCollectionData;
    }

    //Get XYDataset line data for visualization
    private XYDataset getLineData(XYSeries lineSeries) {
        XYSeriesCollection xySeriesCollectionLine = new XYSeriesCollection();
        xySeriesCollectionLine.addSeries(lineSeries);
        return xySeriesCollectionLine;
    }

    //load the data from the txt file into an array
    private static void loadData() {
        try {
            FileReader reader = new FileReader(file);
            BufferedReader bufferedReader = new BufferedReader(reader);
            String line;
            data = new double[dataRow][dataCols];
            String[] columns;
            int i = 0;
            while ((line = bufferedReader.readLine()) != null) {
                columns = line.split(",");
                n = columns.length;
                data[i][0] = Double.parseDouble(columns[0]);
                data[i][1] = Double.parseDouble(columns[1]);
                xAxisValues.add(Double.parseDouble(columns[0]));
                yAxisValues.add(Double.parseDouble(columns[1]));
                i++;
            }
            m = xAxisValues.size();
            bufferedReader.close();
            reader.close();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    //create matrix X (variables with ones added to the first column)
    private static void createMatrixX() {
        variablesWithOneArray = new double[97][2];
        for (int i = 0; i < dataRow; i++) {
            variablesWithOneArray[i][0] = 1;
            variablesWithOneArray[i][1] = data[i][0];
        }
        X = new Array2DRowRealMatrix(variablesWithOneArray);
    }

    //create vector y (values)
    private static void createVectorY() {
        valuesArray = new double[97];
        for (int i = 0; i < dataRow; i++) {
            valuesArray[i] = data[i][1];
        }
        y = new ArrayRealVector(valuesArray);
    }

    //initiate theta with [0;0]
    private static void createTheta() {
        theta = new ArrayRealVector(new double[]{0, 0});
    }

    private static void createCostFunction() {
        RealVector h = X.operate(theta);  // h = X*theta
        RealVector sqrErrors = h.subtract(y).ebeMultiply(h.subtract(y)); //qrErrors = (h-y).^2;
        double[] temp = sqrErrors.toArray();
        double sumOfsqrErrors = DoubleStream.of(temp).sum();
        double J = (1 / (2 * m)) * sumOfsqrErrors;
        System.out.println("Cost function value with theta [0;0] : " + J);
    }

    private static boolean doGradientDescent() {
        RealVector delta = ((X.transpose().multiply(X).operate(theta)).subtract(X.transpose().operate(y))).mapMultiply(1 / m); // delta=1/m*(X'*X*theta-X'*y)
        theta = theta.subtract(delta.mapMultiply(alpha)); // theta=theta-alpha.*delta
        if (iterationsVar == iterations) {
            System.out.println("theta after 1500 iterations: " + theta);
            System.out.println("h = " + theta.getEntry(0) + " + " + theta.getEntry(1) + "x");
            return true;
        }
        iterationsVar++;
        createCostFunction();
        return doGradientDescent();
    }

}

