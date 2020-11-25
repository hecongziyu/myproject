import java.io.*;
import java.util.Vector;
import java.util.Stack;

public class GenerateCode
{
	int INPUT_COLUMNS;
	int graph[][];
	String names[];


	public GenerateCode(String inputFile)
	{
		readFile(inputFile);
		outputStructureMathematica();
	}


	private void readFile(String inputFile)
	{
		//get array sizes
		try
		{
			String line = "";
            
			BufferedReader br = new BufferedReader(new FileReader(inputFile));
			
			line = br.readLine();
			String sizeArray[] = line.split(",");
            
			INPUT_COLUMNS = sizeArray.length;
            
			br.close();
            
		}
		catch(Exception e)
		{
			e.printStackTrace();
			System.exit(-1);
		}

		//////////////////////////////////////////////////////////

		
		graph = new int[INPUT_COLUMNS][INPUT_COLUMNS];
		names = new String[INPUT_COLUMNS];

		//initialize everything to -1
		for(int i=0; i<INPUT_COLUMNS; i++)
		{
			for(int j=0; j<INPUT_COLUMNS; j++)
			{
				graph[i][j] = -1;
			}
		}

		try
		{
			String line = "";
			BufferedReader br = new BufferedReader(new FileReader(inputFile));
			
			int currentRow = 0;

			line = br.readLine();
			names = line.split(",");
            
			while((line = br.readLine()) != null)
			{
				String row[] = line.split(",");
                
				for(int i=0; i<INPUT_COLUMNS; i++)
				{
					graph[currentRow][i] = Integer.parseInt(row[i]);
				}
                
				currentRow++;
			}
            
			br.close();            
		}
		catch(Exception e)
		{
			e.printStackTrace();
			System.exit(-1);
		}


		///////////////////////////////////////////////////////
        
		//check to make sure everything was loaded okay
		for(int i=0; i<INPUT_COLUMNS; i++)
		{
			for(int j=0; j<INPUT_COLUMNS; j++)
			{
				if(graph[i][j] == -1)
				{
					System.out.println("Input Structure file not loaded correctly");
					System.exit(-1);
				}
			}
		}
	}


	private void outputStructureMathematica()
	{
		StringBuilder outputString = new StringBuilder(INPUT_COLUMNS * INPUT_COLUMNS);

		outputString = outputString.append("LayeredGraphPlot[");
		outputString = outputString.append("{");
       
		for(int i=0; i<INPUT_COLUMNS; i++)
		{
			for(int j=0; j<INPUT_COLUMNS; j++)
			{
				if(graph[i][j] == 1)
				{
					outputString = outputString.append(names[i]);
					outputString = outputString.append("->");
					outputString = outputString.append(names[j]);
					outputString = outputString.append(", ");

				}
			}
		}

			
		outputString = outputString.deleteCharAt(outputString.length()-1);
		outputString = outputString.deleteCharAt(outputString.length()-1);
		outputString = outputString.append("}, ");
		outputString = outputString.append("VertexLabeling -> True]");
		outputString = outputString.append("\n");     
			
		

		String dataString = outputString.toString();

		try
		{
			FileWriter file = new FileWriter("Code.txt", false);
            
			file.write(dataString);				
			file.flush();
			file.close();
		}
		catch(Exception e)
		{
			e.printStackTrace();
			System.exit(0);
		}
	}

	//

	public static void main(String args[])
	{
		if(args.length < 1)
		{
			System.out.println("Must enter an input file");
		}

		GenerateCode gc = new GenerateCode(args[0]);
	}
}

