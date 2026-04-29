#@ IOService ioService
#@ UIService uiService
#@ CommandService commandService
#@ BatchService batchService
#@ DefaultDisplayService defaultDisplayService
#@ LogService logService
#@ File(style="directory") inputDir  // This opens a directory chooser instead of a file chooser

import ij.*
import net.imagej.Dataset
import org.scijava.table.DefaultGenericTable
import org.scijava.command.CommandModule
import java.util.concurrent.Future
import java.util.HashMap
import java.lang.Object

// Find the correct class name for the Generalized Dimension command
def commandInfo = commandService.getCommand("at.csa.csaj.plugin2d.frac.Csaj2DFracDimCorrelationCmd")
if (commandInfo == null) {
    println("Looking for alternative command class names...")
    def allCommands = commandService.getCommandsOfType(org.scijava.command.Command.class)
    def possibleCommands = allCommands.findAll { it.getClassName().toLowerCase().contains("general") && it.getClassName().toLowerCase().contains("dim") }
    
    if (possibleCommands.size() > 0) {
        commandInfo = possibleCommands[0]
        println("Found possible command: " + commandInfo.getClassName())
    } else {
        println("ERROR: Could not find Generalized Dimension command class.")
        println("Available fractal dimension commands:")
        allCommands.findAll { it.getClassName().toLowerCase().contains("frac") && it.getClassName().toLowerCase().contains("dim") }.each {
            println("  - " + it.getClassName())
        }
        return
    }
}

// Get the parameters accepted by the command to avoid using invalid ones
def inputs = commandInfo.inputs()
println("Available command parameters:")
inputs.each { param ->
    println("  - " + param.getName())
}

// Create a table to store all results
def resultsTable = new DefaultGenericTable()
resultsTable.appendColumn("File name")
resultsTable.appendColumn("Slice name")
resultsTable.appendColumn("# Boxes")   
resultsTable.appendColumn("Reg Start")
resultsTable.appendColumn("Reg End")
resultsTable.appendColumn("Scanning type")
resultsTable.appendColumn("Color model")
resultsTable.appendColumn("(Sliding disc) Pixel %")
resultsTable.appendColumn("Dc") 
resultsTable.appendColumn("R2")   
resultsTable.appendColumn("StdErr")  

// Get all image files in the directory
def imageFiles = inputDir.listFiles().findAll { file ->
    // Filter files with common image extensions
    def name = file.getName().toLowerCase()
    name.endsWith('.tif') || name.endsWith('.tiff') || 
    name.endsWith('.jpg') || name.endsWith('.jpeg') || 
    name.endsWith('.png') || name.endsWith('.gif') ||
    name.endsWith('.bmp')
}

println("Found ${imageFiles.size()} image files to process")

// Process each image file
imageFiles.eachWithIndex { file, index ->
    println("Processing ${index+1}/${imageFiles.size()}: ${file.getName()}")
    
    try {
        // Load the dataset
        Dataset datasetIn = (Dataset)ioService.open(file.getPath())
        
        // Configure parameters for the generalized dimension calculation
        HashMap<String, Object> parameters = new HashMap<String, Object>()
        parameters.put("datasetIn", datasetIn)
        parameters.put("spinnerInteger_NumBoxes", "11")         // Number of radii
        parameters.put("spinnerInteger_NumRegStart", "1")      // Regression start = 1
        parameters.put("spinnerInteger_NumRegEnd", "11")        // Regression end
        parameters.put("choiceRadioButt_ScanningType", "Raster box")  // Scanning - Raster Box
        parameters.put("choiceRadioButt_ColorModelType", "Binary")      // Color Model - Binary
        parameters.put("spinnerInteger_PixelPercentage", "10")         // Pixel % = 10
        parameters.put("booleanShowDoubleLogPlot", "false")    // Show double log plot = false
        parameters.put("booleanOverwriteDisplays", "true")
        parameters.put("spinnerInteger_NumImageSlice", "1")    //always first image
        
        // Run the command and wait for it to complete
        Future future = commandService.run(commandInfo.getClassName(), false, parameters)
        CommandModule commandModule = future.get() // block till complete
        
        // Get the results table
        DefaultGenericTable tableOut = (DefaultGenericTable)commandModule.getOutput("tableOut")
        
        if (tableOut == null) {
            println("Warning: No output table returned for ${file.getName()}")
            println("Available outputs:")
            commandModule.getOutputs().each { name, value ->
                println("  - ${name}: ${value != null ? value.getClass().getName() : 'null'}")
            }
            //continue
        }
        
        // Debug: Print table structure
        println("Output table structure:")
        for (int col = 0; col < tableOut.getColumnCount(); col++) {
            println("  Column ${col}: ${tableOut.getColumnHeader(col)}")
        }
        println("  #Rows: ${tableOut.getRowCount()}")
        
        // Print the first few rows for debugging
        for (int r = 0; r < Math.min(5, tableOut.getRowCount()); r++) {
            StringBuilder rowStr = new StringBuilder("  Row ${r}: ")
            for (int c = 0; c < tableOut.getColumnCount(); c++) {
                rowStr.append("${tableOut.get(c, r)}, ")
            }
            println(rowStr.toString())
        }
        
        // Add the result to our table
        resultsTable.appendRow()
        int lastRow = resultsTable.getRowCount() - 1
        resultsTable.set(0,  lastRow, file.getName())    
        resultsTable.set(1,  lastRow, tableOut.get(1, 0))
        resultsTable.set(2,  lastRow, tableOut.get(2, 0))
        resultsTable.set(3,  lastRow, tableOut.get(3, 0))
        resultsTable.set(4,  lastRow, tableOut.get(4, 0))
        resultsTable.set(5,  lastRow, tableOut.get(5, 0))
        resultsTable.set(6,  lastRow, tableOut.get(6, 0))
        resultsTable.set(7,  lastRow, tableOut.get(7, 0))
        resultsTable.set(8,  lastRow, tableOut.get(8, 0))
        resultsTable.set(9,  lastRow, tableOut.get(9, 0))
        resultsTable.set(10,  lastRow, tableOut.get(10, 0))
    
    } catch (Exception e) {
        println("Error processing ${file.getName()}: ${e.getMessage()}")
        e.printStackTrace()
    }
}

// Show the results table
if (resultsTable.getRowCount() > 0) {
    uiService.show("2D FracDimCorrelation Results", resultsTable)
    
    // Save the results as CSV file
    try {
        def outputFile = new File(inputDir, "correlation_dimension_results.csv")
        def writer = new PrintWriter(outputFile)
        
        // Write header
        List<String> headers = []
        for (int col = 0; col < resultsTable.getColumnCount(); col++) {
            headers.add(resultsTable.getColumnHeader(col))
        }
        writer.println(headers.join(","))
        
        // Write data rows
        for (int row = 0; row < resultsTable.getRowCount(); row++) {
            List<String> rowValues = []
            for (int col = 0; col < resultsTable.getColumnCount(); col++) {
                Object value = resultsTable.get(col, row)
                rowValues.add(value != null ? value.toString() : "")
            }
            writer.println(rowValues.join(","))
        }
        
        writer.close()
        println("Results saved to CSV: ${outputFile.getAbsolutePath()}")
    } catch (Exception e) {
        println("Error saving CSV file: ${e.getMessage()}")
        e.printStackTrace()
    }
}

println("Processing complete!")