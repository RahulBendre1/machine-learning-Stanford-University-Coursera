package com.mkis.assignments.neuralnetwork;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;

import java.util.List;

/**Plotting the final cost function of the training and cross-validation set to see if high variance (over-fitting) or bias (under-fitting) occurs.
 * Choose lambda where the Cross-Validation set's cost function has its minimum.
 */

public class Evaluation extends ApplicationFrame {

    Evaluation(final String title, List<Double> trainingSetCostF, List<Double> CVSetSetCostF, double[] lambda) {

        super(title);

        XYSeriesCollection seriesCollection = new XYSeriesCollection();

        // Create series for the training set's cost function
        XYSeries lineSeriesTrSet = new XYSeries("Training set");
        for (int i = 0; i < lambda.length; i++) {
            lineSeriesTrSet.add(lambda[i], trainingSetCostF.get(i));
        }
        seriesCollection.addSeries(lineSeriesTrSet);

        // Create series for the cross-validation set's cost function
        XYSeries lineSeriesCVSet = new XYSeries("Cross validation set");
        for (int i = 0; i < lambda.length; i++) {
            lineSeriesCVSet.add(lambda[i], CVSetSetCostF.get(i));
        }
        seriesCollection.addSeries(lineSeriesCVSet);

        // Create the chart with the plot and a legend
        JFreeChart chart = ChartFactory.createXYLineChart("Evaluation", "lambda", "Cost function", seriesCollection, PlotOrientation.VERTICAL, true, true, false);

        XYPlot plot = (XYPlot) chart.getPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();

        //Training set
        renderer.setSeriesLinesVisible(0, true);
        renderer.setSeriesShapesVisible(0, true);
        //Cross-validation set
        renderer.setSeriesLinesVisible(1, true);
        renderer.setSeriesShapesVisible(1, true);

        plot.setRenderer(renderer);
        final ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(800, 500));
        setContentPane(chartPanel);
    }

}
