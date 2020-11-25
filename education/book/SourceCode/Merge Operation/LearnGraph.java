import java.io.*;
import java.util.Vector;
import java.util.Stack;
import java.util.Random;


public class LearnGraph
{
	private Vector <int[]> groupVector;
	private Vector <int[][]> structureVector;

	private int structureCopy[][];
	private int groupArrayCopy[];

	private int structure[][];
	private int groupArray[];
	private int INPUT_ROWS, INPUT_COLUMNS;

	String skillNames[];

	public LearnGraph(String structurePath, String groupPath)
	{
		groupVector = new Vector();
		structureVector = new Vector();

		readFile(structurePath, groupPath);
		outputMerges();

		for(int i=0; i<100000; i++)
		{
			//copy stuff then run off of copy

			structureCopy = new int[INPUT_COLUMNS][INPUT_COLUMNS];
			groupArrayCopy = new int[INPUT_COLUMNS];

			for(int j=0; j<INPUT_COLUMNS; j++)
			{
				for(int k=0; k<INPUT_COLUMNS; k++)
				{
					structureCopy[j][k] = structure[j][k];
				}

				groupArrayCopy[j] = groupArray[j];
			}

			runCrushing();

			for(int j=0; j<INPUT_COLUMNS; j++)
			{
				structureCopy[j][j] = 0;
			}

			//save copy if not already exists and acyclic

			boolean acyclic = cycleCheck(structureCopy, groupArrayCopy);


			//check all in order...

			boolean exists = false;

			for(int h=0; h<groupVector.size() && !exists; h++)
			{
				int tempGroup[] = groupVector.get(h);
				int tempStructure[][] = structureVector.get(h);


				boolean same = true;

				for(int j=0; j<INPUT_COLUMNS && same; j++)
				{
					if(tempGroup[j] != groupArrayCopy[j])
					{
						same = false;
					}
				}

				if(same)
				{
					for(int j=0; j<INPUT_COLUMNS && same; j++)
					{
						for(int k=0; k<INPUT_COLUMNS && same; k++)
						{
							if(tempStructure[j][k] != structureCopy[j][k])
							{
								same = false;
							}
						}
					}
				}

				if(same)
				{
					exists = true;
				}
			}

			if(!exists && acyclic)
			{
				groupVector.add(groupArrayCopy);
				structureVector.add(structureCopy);
			}

			System.out.println(i);
		}

		for(int i=0; i<groupVector.size(); i++)
		{
			outputStructureFirstRowHeader(i, structureVector.get(i), groupVector.get(i));
			outputStructureNoHeaders(i, structureVector.get(i), groupVector.get(i));
			outputStructureMatlab(i, structureVector.get(i), groupVector.get(i));
		}

		outputStructureMathematica();
		

		System.out.println("XX" + groupVector.size());
	}

	private void outputMerges()
	{
		for(int i=0; i<INPUT_COLUMNS; i++)
		{
			for(int j=i+1; j<INPUT_COLUMNS; j++)
			{
				if(structure[i][j] == 1)
				{
					System.out.println("Possible Merge = Skill " + skillNames[i] + " to skill " + skillNames[j]);
				}
			}
		}
	}

	private void readFile(String structurePath, String groupPath)
	{
		//get array sizes
		try
		{
			String line = "";
            
			BufferedReader br = new BufferedReader(new FileReader(structurePath));
			
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

		
		structure = new int[INPUT_COLUMNS][INPUT_COLUMNS];

		//initialize everything to -1
		for(int i=0; i<INPUT_COLUMNS; i++)
		{
			for(int j=0; j<INPUT_COLUMNS; j++)
			{
				structure[i][j] = -1;
			}
		}

		try
		{
			String line = "";
			BufferedReader br = new BufferedReader(new FileReader(structurePath));
			
			int currentRow = 0;
            
			while((line = br.readLine()) != null)
			{
				String row[] = line.split(",");
                
				for(int i=0; i<INPUT_COLUMNS; i++)
				{
					structure[currentRow][i] = Integer.parseInt(row[i]);
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


		////////////////////////////

		groupArray = new int[INPUT_COLUMNS];

		//initialize everything to -1
		for(int i=0; i<INPUT_COLUMNS; i++)
		{
			groupArray[i] = -1;
		}

		try
		{
			
			String line = "";
            
			BufferedReader br = new BufferedReader(new FileReader(groupPath));
			
			line = br.readLine();
			skillNames = line.split(",");
            
			line = br.readLine();
			String groupArray0[] = line.split(",");

			br.close();

			for(int i=0; i<skillNames.length; i++)
			{
				groupArray[i] = Integer.parseInt(groupArray0[i]);
			}
            
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
				if(structure[i][j] == -1)
				{
					System.out.println("Input Structure file not loaded correctly");
					System.exit(-1);
				}
			}

			if(groupArray[i] == -1)
			{
				System.out.println("Input groups file not loaded correctly");
				System.exit(-1);
			}
		}
	}

	private boolean cycleCheck(int structures[][], int groups[])
	{
		boolean rv = true;

		Vector <Integer> uniqueGroups = new Vector();

		for(int i=0; i<groups.length; i++)
		{
			if(!uniqueGroups.contains(groups[i]))
			{
				uniqueGroups.add(groups[i]);
			}
		}

		int skillStructure[][] = new int[uniqueGroups.size()][uniqueGroups.size()];

		int rowCount = 0;
		int columnCount = 0;

		for(int i=0; i<INPUT_COLUMNS; i++)
		{
			boolean iMember = false;

			for(int j=0; j<INPUT_COLUMNS; j++)
			{
				boolean jMember = false;

				for(int k=0; k<groups.length; k++)
				{
					if(i == groups[k])
					{
						iMember = true;
					}
					
					if(j == groups[k])
					{
						jMember = true;
					}
				}

				//i and j are both group numbers
				if(iMember && jMember)
				{					
					skillStructure[rowCount][columnCount] = structures[i][j];
					columnCount++;
				}
			}
            
			if(iMember)
			{
				rowCount++;
				columnCount = 0;
			}
		}

		Stack <Integer> removed = new Stack();
		boolean change = true;

		while(change)
		{
			change = false;

			for(int i=0; i<uniqueGroups.size() && !change; i++)
			{
				boolean incoming = false;

				for(int j=0; j<uniqueGroups.size(); j++)
				{
					//detect incoming edge
					if(skillStructure[j][i] == 1)
					{
						incoming = true;
					}
				}

				if(!incoming && removed.search(i) < 0)
				{
					removed.push(i);

					for(int j=0; j<uniqueGroups.size(); j++)
					{
						skillStructure[i][j] = 0;
					}

					change = true;
				}
			}
		}

		if(removed.size() != uniqueGroups.size())
		{
			rv = false;
		}
		
		return rv;
	}

	private void runCrushing()
	{
		boolean keepRunning = true;
		int iteration = 0;

		Random randomNumber = new Random();

		while(iteration <= 13 /*keepRunning*/)
		{
			
				//pick two connected groups somewhat randomly

				int group1 = -1;
				int group2 = -1;

				if(iteration == 0)
				{
					group1 = 21/*first group combined*/;
					group2 = -1;
				}
				else if(iteration == 1)
				{
					group1 = 9;
					group2 = -1;
				}
				else if(iteration == 2)
				{
					group1 = 9;
					group2 = -1;
				}
				else if(iteration == 3)
				{
					group1 = 16;
					group2 = -1;
				}
				else if(iteration == 4)
				{
					group1 = 13;
					group2 = -1;
				}
				else if(iteration == 5)
				{
					group1 = 9;
					group2 = -1;
				}
				else if(iteration == 6)
				{
					group1 = 8;
					group2 = -1;
				}
				else if(iteration == 7)
				{
					group1 = 2;
					group2 = -1;
				}
				else if(iteration == 8)
				{
					group1 = 8;
					group2 = -1;
				}
				else if(iteration == 9)
				{
					group1 = 18;
					group2 = -1;
				}
				else if(iteration == 10)
				{
					group1 = 8;
					group2 = -1;
				}
				else if(iteration == 11)
				{
					group1 = 0;
					group2 = -1;
				}
				else if(iteration == 12)
				{
					group1 = 8;
					group2 = -1;
				}
				/*else if(iteration == 13)
				{
					//group1 = 16;
					//group2 = -1;
				}
				*/
				else
				{

					//question spot, not actually the group, need to use as array spot for group number
					group1 = randomNumber.nextInt(groupArrayCopy.length);
					group2 = -1;

				}

				boolean adjacent = false;

				while(!adjacent)
				{
					if(iteration == 0)
					{
						group2 = 23/*second group to combine*/;
					}
					else if(iteration == 1)
					{
						group2 = 10;
					}
					else if(iteration == 2)
					{
						group2 = 11;
					}
					else if(iteration == 3)
					{
						group2 = 19;
					}
					else if(iteration == 4)
					{
						group2 = 15;
					}
					else if(iteration == 5)
					{
						group2 = 16;
					}
					else if(iteration == 6)
					{
						group2 = 9;
					}
					else if(iteration == 7)
					{
						group2 = 6;
					}
					else if(iteration == 8)
					{
						group2 = 14;
					}
					else if(iteration == 9)
					{
						group2 = 21;
					}
					else if(iteration == 10)
					{
						group2 = 13;
					}
					else if(iteration == 11)
					{
						group2 = 2;
					}
					else if(iteration == 12)
					{
						group2 = 18;
					}
					/*else if(iteration == 13)
					{
						//group2 = 19;
					}*/
					else
					{
						group2 = randomNumber.nextInt(groupArrayCopy.length);
					}

					if(group2 !=  group1 && groupArrayCopy[group2] != groupArrayCopy[group1] && (structureCopy[group2][group1] == 1 || structureCopy[group1][group2] == 1))
					{
						adjacent = true;
					}
				}

				boolean done = false;

				for(int i=0; i<groupArrayCopy.length && !done; i++)
				{
					if(groupArrayCopy[i] == groupArrayCopy[group1])
					{
						crush(group1, group2);
						done = true;
					}

					if(groupArrayCopy[i] == groupArrayCopy[group2])
					{
						crush(group2, group1);
						done = true;
					}
				}

				//determine keepRunning randomly from iteration

				iteration++;

				int stop = randomNumber.nextInt(groupArrayCopy.length - iteration);

				if(stop == 0)
				{
					keepRunning = false;
				}
			
			//make sure there is more than 1 group

			if(keepRunning)
			{
				keepRunning = false;

				for(int i=0; i<groupArrayCopy.length; i++)
				{
					if(groupArrayCopy[i] != 0)
					{
						keepRunning = true;
					}
				}
			}			
		}
	}

	private void crush(int question1, int question2)
	{
		int oldGroup = groupArrayCopy[question2];

		for(int i=0; i<INPUT_COLUMNS; i++)
		{
			if(groupArrayCopy[i] == oldGroup)
			{
				groupArrayCopy[i] = groupArrayCopy[question1];
			}
		}

		Vector <Integer> groupNumbers = new Vector();

		for(int i=0; i<groupArrayCopy.length; i++)
		{
			if(groupArrayCopy[i] == groupArrayCopy[question1])
			{
				groupNumbers.add(i);
			}
		}

		//for all group members, all same children
		//for all group members, columns should be the same		
		for(int i=0; i<groupNumbers.size(); i++)
		{
			for(int j=0; j<groupArrayCopy.length; j++)
			{
				if(structureCopy[groupNumbers.get(i)][j] == 1)
				{
					for(int k=0; k<groupNumbers.size(); k++)
					{
						structureCopy[groupNumbers.get(k)][j] = 1;
					}
				}

				if(structureCopy[j][groupNumbers.get(i)] == 1)
				{
					for(int k=0; k<groupNumbers.size(); k++)
					{
						structureCopy[j][groupNumbers.get(k)] = 1;
					}
				}
			}
		}
	}

	///////////////////////////////////////////////////////////

	private void outputStructureFirstRowHeader(int outputNumber, int structures[][], int groups[])
	{
		StringBuilder outputString = new StringBuilder(INPUT_COLUMNS * INPUT_COLUMNS);

		Vector <Integer> uniqueGroups = new Vector();

		for(int i=0; i<groups.length; i++)
		{
			if(!uniqueGroups.contains(groups[i]))
			{
				uniqueGroups.add(groups[i]);
			}
		}

		for(int i=0; i<uniqueGroups.size(); i++)
		{
			for(int j=0; j<groups.length; j++)
			{
				if(groups[j] == uniqueGroups.get(i))
				{
					outputString = outputString.append(j+1);
					outputString = outputString.append(" ; ");
				}
			}

			outputString = outputString.append(",");
		}

		outputString = outputString.append("\n");
       
		for(int i=0; i<INPUT_COLUMNS; i++)
		{
			boolean iMember = false;

			for(int j=0; j<INPUT_COLUMNS; j++)
			{
				boolean jMember = false;

				for(int k=0; k<groups.length; k++)
				{
					if(i == groups[k])
					{
						iMember =true;
					}
					
					if(j == groups[k])
					{
						jMember = true;
					}
				}

				//i and j are both group numbers
				if(iMember && jMember)
				{					
					outputString = outputString.append(Integer.toString(structures[i][j]));
					outputString = outputString.append(",");
				}
			}
            
			if(iMember)
			{
				outputString = outputString.append("\n");
			}
		}
        
		String dataString = outputString.toString();
        
		try
		{
			FileWriter file = new FileWriter("OutputFirstRowHeader" + File.separator +"OutputStructure" + (outputNumber+1) + ".csv", false);
            
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

	private void outputStructureNoHeaders(int outputNumber, int structures[][], int groups[])
	{
		StringBuilder outputString = new StringBuilder(INPUT_COLUMNS * INPUT_COLUMNS);
       
		for(int i=0; i<INPUT_COLUMNS; i++)
		{
			boolean iMember = false;

			for(int j=0; j<INPUT_COLUMNS; j++)
			{
				boolean jMember = false;

				for(int k=0; k<groups.length; k++)
				{
					if(i == groups[k])
					{
						iMember =true;
					}
					
					if(j == groups[k])
					{
						jMember = true;
					}
				}

				//i and j are both group numbers
				if(iMember && jMember)
				{					
					outputString = outputString.append(Integer.toString(structures[i][j]));
					outputString = outputString.append(",");
				}
			}
            
			if(iMember)
			{
				outputString = outputString.append("\n");
			}
		}
        
		String dataString = outputString.toString();
        
		try
		{
			FileWriter file = new FileWriter("OutputNoHeaders" + File.separator +"Structure" + (outputNumber+1) + ".csv", false);
            
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

	//sturctureVector + groupVector

	private void outputStructureMathematica()
	{
		StringBuilder outputString = new StringBuilder(INPUT_COLUMNS * INPUT_COLUMNS * structureVector.size());

		for(int h=0; h<structureVector.size(); h++)
		{
			outputString = outputString.append("(*");
			outputString = outputString.append(h+1);
			outputString = outputString.append("*)");

			outputString = outputString.append("LayeredGraphPlot[");
			outputString = outputString.append("{");
       

			int structures[][] = structureVector.get(h);
			int groups[] = groupVector.get(h);

			Vector <Integer> uniqueGroups = new Vector();

			for(int i=0; i<groups.length; i++)
			{
				if(!uniqueGroups.contains(groups[i]))
				{
					uniqueGroups.add(groups[i]);
				}
			}

			//create the group name by combining all skills in the group into one name

			String skillGroupNames[] = new String[uniqueGroups.size()];

			for(int i=0; i<uniqueGroups.size(); i++)
			{
				String name = "";

				for(int j=0; j<groups.length; j++)
				{
					String skillName = skillNames[j];

					if(groups[j] == uniqueGroups.get(i) && name.indexOf(skillName) == -1)
					{
						name = name.concat(skillName + "x");
					}
				}

				skillGroupNames[i] = name.substring(0, name.length()-1);

				//System.out.println(name);
			}


			for(int i=0; i<uniqueGroups.size(); i++)
			{
				int currentGroup = uniqueGroups.get(i);
				int firstGroupMember = -1;

				boolean found = false;

				for(int j=0; j<groups.length && !found; j++)
				{
					if(groups[j] == currentGroup)
					{
						firstGroupMember = j;
						found = true;
					}
				}

				//get all unique groups that this current group points to

				Vector <Integer> uniqueGroups2 = new Vector();

				for(int j=0; j<groups.length; j++)
				{
					if(groups[firstGroupMember] != groups[j] && structures[firstGroupMember][j] == 1)
					{
						if(!uniqueGroups2.contains(groups[j]))
						{
							uniqueGroups2.add(groups[j]);
						}
					}
				}

				for(int j=0; j<uniqueGroups2.size(); j++)
				{
					int groupNum = uniqueGroups2.get(j);

					//find skill group name that has skill name of the group number

					int specialSpot = -1;

					for(int k=0; k<skillGroupNames.length && specialSpot == -1; k++)
					{
						if(skillGroupNames[k].indexOf(skillNames[groupNum]) != -1)
						{
							specialSpot = k;
						}
					}

					outputString = outputString.append(skillGroupNames[i] + " -> " + skillGroupNames[specialSpot] + ", ");
				}

				for(int j=0; j<groups.length; j++)
				{
					if(groups[j] == groups[currentGroup])
					{
						int specialSpot = -1;

						for(int k=0; k<skillGroupNames.length && specialSpot == -1; k++)
						{
							if(skillGroupNames[k].indexOf(skillNames[currentGroup]) != -1)
							{
								specialSpot = k;
							}
						}

						outputString = outputString.append(skillGroupNames[specialSpot] + "-> I" + (j+1) + ", ");
					}
				}
			}

			outputString = outputString.deleteCharAt(outputString.length()-1);
			outputString = outputString.deleteCharAt(outputString.length()-1);
			outputString = outputString.append("}, ");
			outputString = outputString.append("VertexLabeling -> True]");
			outputString = outputString.append("\n");     
			
		}

		String dataString = outputString.toString();

		try
		{
			FileWriter file = new FileWriter("OutputMathematica" + File.separator +"Structures.txt", false);
            
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

	private void outputStructureMatlab(int outputNumber, int structures[][], int groups[])
	{
		StringBuilder outputString = new StringBuilder((INPUT_COLUMNS * INPUT_COLUMNS) + 10000);

		outputString = outputString.append("function dag = structure_" + (outputNumber+1) + "\n");
		outputString = outputString.append("% auto-generated code by Doug based on Zachs code" + "\n");
		outputString = outputString.append("% Bayesian Prediction Analysis for Dynamic Learning Maps (Kansas)" + "\n");
		outputString = outputString.append("% in collaboration with Neil Heffernan and colleagues  (WPI)" + "\n");
		outputString = outputString.append("% data property of Angela Broaddus (Kansas) and Neal Kingston (Kansas)" + "\n");

		Vector <Integer> uniqueGroups = new Vector();

		for(int i=0; i<groups.length; i++)
		{
			if(!uniqueGroups.contains(groups[i]))
			{
				uniqueGroups.add(groups[i]);
			}
		}


		int nSize = uniqueGroups.size() + INPUT_COLUMNS;
		int nCount = 1;

		outputString = outputString.append("N=" + nSize + ";\n" + "\n");


		//create the group name by combining all skills in the group into one name

		String skillGroupNames[] = new String[uniqueGroups.size()];

		for(int i=0; i<uniqueGroups.size(); i++)
		{
			String name = "";

			for(int j=0; j<groups.length; j++)
			{
				String skillName = skillNames[j];

				if(groups[j] == uniqueGroups.get(i) && name.indexOf(skillName) == -1)
				{
					name = name.concat(skillName + "_");
				}
			}

			skillGroupNames[i] = name.substring(0, name.length()-1);
		}

		//output groups and questions

		for(int i=0; i<skillGroupNames.length; i++)
		{
			outputString = outputString.append(skillGroupNames[i]);
			outputString = outputString.append("=" + nCount + ";\n");
			nCount++;
		}

		for(int i=0; i<INPUT_COLUMNS; i++)		
		{
			outputString = outputString.append("I" + (i+1) + "=" + nCount + ";\n");
			nCount++;
		}

		outputString = outputString.append("\ndag=zeros(N,N); \n");

		//output links

		for(int i=0; i<uniqueGroups.size(); i++)
		{
			outputString = outputString.append("dag(");

			int currentGroup = uniqueGroups.get(i);
			int firstGroupMember = -1;

			boolean found = false;

			for(int j=0; j<groups.length && !found; j++)
			{
				if(groups[j] == currentGroup)
				{
					firstGroupMember = j;
					found = true;
				}
			}

			//get all unique groups that this current group points to

			Vector <Integer> uniqueGroups2 = new Vector();

			for(int j=0; j<groups.length; j++)
			{
				if(groups[firstGroupMember] != groups[j] && structures[firstGroupMember][j] == 1)
				{
					if(!uniqueGroups2.contains(groups[j]))
					{
						uniqueGroups2.add(groups[j]);
					}
				}
			}

			outputString = outputString.append(skillGroupNames[i] + ", [");

			int count = 0;

			for(int j=0; j<uniqueGroups2.size(); j++)
			{
				int groupNum = uniqueGroups2.get(j);

				//find skill group name that has skill name of the group number

				int specialSpot = -1;

				for(int k=0; k<skillGroupNames.length && specialSpot == -1; k++)
				{
					if(skillGroupNames[k].indexOf(skillNames[groupNum]) != -1)
					{
						specialSpot = k;
					}
				}

				outputString = outputString.append(skillGroupNames[specialSpot] + " ");

				count++;
			}

			if(uniqueGroups.size() == 1)
			{
				outputString = outputString.append(skillGroupNames[0] + ", [");
			}

			//attach questions too


			for(int j=0; j<groups.length; j++)
			{
				if(groups[j] == groups[currentGroup])
				{
					outputString = outputString.append("I" + (j+1) + " ");
					count++;
				}
			}


			outputString = outputString.delete(outputString.length()-1, outputString.length());
			outputString = outputString.append("]) = 1;" + "\n");

			if(count <= 1)
			{
				int indexOfOpen = outputString.lastIndexOf("[");
				int indexOfClosed = outputString.lastIndexOf("]") - 1;
			
				outputString = outputString.deleteCharAt(indexOfOpen);
				outputString = outputString.deleteCharAt(indexOfClosed);
			}
		}
        


		String dataString = outputString.toString();
        
		try
		{
			FileWriter file = new FileWriter("OutputMatLab" + File.separator +"OutputStructure" + "_"  + uniqueGroups.size() + "_" + (outputNumber+1) + ".m", false);
            
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

	public static void main(String args[])
	{
		if(args.length < 2)
		{
			System.out.println("Must enter an input structure files");
		}

		LearnGraph lg = new LearnGraph(args[0], args[1]);
	}
}