package weka.classifiers.trees;


import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

public class FilterTree extends AbstractClassifier {

    /** for serialization */
    //static final long serialVersionUID = -251831442047263433L;

    /** The filter to use. */
    protected Filter m_Filter = new Randomize();

    /** The minimum number of instances used to continue growing the tree */
    protected int m_minInstances = 100;

    /** The root node of the tree**/
    protected Node rootNode;

    /** A list of instances that can be referenced for retrievial of an instance **/
    protected Instances GlobalInstances;

    /** Store computated sorted index of features **/
    protected int sortedFeaturesIndex[][];

    /**
     * Returns the Capabilities of this classifier.
     *
     * @return the capabilities of this object
     * @see Capabilities
     */
    @Override
    public Capabilities getCapabilities() {

        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.BINARY_CLASS);
        //result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        return result;


    }

    /** Handling the Filter parameter. */
    @OptionMetadata(
            displayName = "Filter function",
            description = "The Filter to use.", displayOrder = 1,
            commandLineParamName = "F",
            commandLineParamSynopsis = "-F <Filter specification>")
    public Filter getFilter() {  return m_Filter; }
    public void setFilter(Filter value) { m_Filter = value; }

    /** Handling the parameter setting the sample size. */
    @OptionMetadata(
            displayName = "The expansion threshold.",
            description = "The minimum number of instances used to continue growing the tree.", displayOrder = 3,
            commandLineParamName = "M",
            commandLineParamSynopsis = "-M <int>")
    public void setExpansionThreshold(int minNumberInstances) { m_minInstances = minNumberInstances; }
    public int getExpansionThreshold() { return m_minInstances; }

    /**
     * An interface indicating objects storing node information, implemented by three node info classes.
     */
    private interface NodeInfo extends Serializable {};

    private class  RootNodeInfo implements NodeInfo{

        // The data to be used for expanding the node.
        protected Instances Data;


        /**
         * Constructs an Root Node object.
         *
         * @param data the data to be used for turning this node into an expanded node.
         */
        public RootNodeInfo(Instances data) {
            Data = data;
        }
    }

    /**
     * Class whose objects represent split nodes.
     */
    private class SplitNodeInfo implements NodeInfo {

        // The left node
        private Node left;

        // The right node
        private Node right;

        // The attribute used for splitting
        protected Attribute SplitAttribute;

        // The split value
        protected double SplitValue;

        //Getter for left
        protected Node getLeft() {return left;}

        //Setter for left
        protected void setLeft(Node node) {
            this.left = node;
            Instances[0] = node;
        }

        //Getter for right
        protected Node getRight() {return right;}

        //Setter for left
        protected void setRight(Node node) {
            this.right = node;
            Instances[1] = node;
        }

        //Pair of instances
        protected Node[] Instances;

        /**
         * Constructs a SplitNodeInfo object
         *
         * @param splitAttribute the attribute that defines the split
         * @param splitValue the value used for the split
         * @param left access to the left side of the tree
         * @param right access to the right side of the tree
         */
        public SplitNodeInfo(Attribute splitAttribute, double splitValue, Node left, Node right) {
            SplitAttribute = splitAttribute;
            SplitValue = splitValue;
            this.left = left;
            this.right = right;
            Instances = new Node[]{left,right};
        }
    }

    /**
     * Class whose objects represent leaf nodes.
     */
    private class LeafNodeInfo implements NodeInfo {

        // The array of predictions
        protected double[] Prediction;

        /**
         * Constructs a LeafNodeInfo object.
         *
         * @param prediction the array of predictions to store at this node
         */
        public LeafNodeInfo(double[] prediction) {
            Prediction = prediction;
        }
    }

    /**
     * Class whose objects represent unexpanded nodes.
     */
    private class UnexpandedNodeInfo implements NodeInfo {

        // The data to be used for expanding the node.
        protected Instances Data;

        /**
         * Constructs an UnexpandedNodeInfo object.
         *
         * @param data the data to be used for turning this node into an expanded node.
         */
        public UnexpandedNodeInfo(Instances data) {
            Data = data;
        }
    }

    /**
     * Class representing a node in the decision tree.
     */
    private class Node implements Serializable {

        // The node information object that stores the actual information for this node.
        protected NodeInfo NodeInfo;

        /**
         * Constructs a node based on the give node info.
         *
         * @param nodeInfo an appropriate node information object
         */
        public Node(FilterTree.NodeInfo nodeInfo) {
            NodeInfo = nodeInfo;
        }
    }


    protected double calculateEntropy(Node node){
        return 0;
    }

    protected Node makeLeafNode(Node node){

        return null;
    }

    /**
     * Method that processes a node. Assumes that the given node is unexpanded. Turns the node
     * into a leaf node or split node as appropriate by replacing the node information.
     *
     * @param node the unexpanded node to process
     * @return the node with updated node information, turning it into a split node or leaf node
     */
    protected Node processNode(Node node) {

        Instances instances = ((UnexpandedNodeInfo)node.NodeInfo).Data;

        //Checking stopping criteria - Certain Number of instances met, as specified by the user
        if(instances.size() <= m_minInstances){
            return makeLeafNode(node);
        }

        //Iterating through each instance and calculating the entropy




        return null;

    }



    public String globalInfo() { return "A tree classifier that local applies a filter at each node."; }



    @Override
    public void buildClassifier(Instances instances) throws Exception {

        //Setting the list of global instances
        GlobalInstances = instances;


        //Creating a node list that has all tree nodes in it before they are finalised.
        ArrayList<Node> nodesInTree = new ArrayList<>();

        //Creating rootNode
        rootNode = new Node(new UnexpandedNodeInfo(instances));

        //Adding node to tree
        nodesInTree.add(rootNode);

        //Splitting node
        while(!nodesInTree.isEmpty()){
            Node newNode = processNode(rootNode);

            //Iterating through node subtrees and expanding to ethier split node or leafnode
            if(newNode.NodeInfo instanceof SplitNodeInfo){
                nodesInTree.addAll(Arrays.asList(((SplitNodeInfo) newNode.NodeInfo).Instances));
            }
        }
    }

    @Override
    public double[][] distributionsForInstances(Instances batch) throws Exception {
        return super.distributionsForInstances(batch);
    }

    public double[] distributionsForInstances(Instance instance) throws Exception {
        return new double[]{0};
    }

    /**
     * The main method used for running this filter from the command-line interface.
     *
     * @param options the command-line options
     */
    public static void main(String[] options) {
        runClassifier(new FilterTree(), options);
    }
}
