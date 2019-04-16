from functools import reduce



def create_tree_rule_filters(X_values, tree):
    """Traverses down Decision Tree paths to extract all possible paths.
    Each path forms a boolean Series, which equals True if its
    corresponding path was taken.

    Parameters
    ----------
    X_values : DataFrame
        The independent variables
    tree : Tree
        A fitted sklearn tree object

    Returns
    -------
    rules_dict : dict
        Maps Decision Tree path (rule) to the boolean Series that
        indicates when the rule fired
    """

    def _traverse(parent_index, rule_path, direction_path):
        """Perform preorder tree traversal and add cumulative rules."""
        # Get the index of the left and right children
        left_child_index = tree.children_left[parent_index]
        right_child_index = tree.children_right[parent_index]

        # Append parent index to the rule path
        rule_path.append(parent_index)

        # TODO: If the root node is split from a binary variable, then
        # it should not be added as a root node. It is redundant since
        # it is the same as an input rule.

        # Do not add path for root node
        #if parent_index != 0:
        if len(rule_path) > 2:
            # Subset on columns that the rule path defines
            X_rule_values = X_values.iloc[:, tree.feature[rule_path]]
            thresholds = tree.threshold[rule_path]

            # Create boolean Series to filter based on rule
            rule_filter_srs =\
                _create_rule_filter(X_rule_values, thresholds, direction_path)
            rule_filter_name =\
                _create_rule_filter_name(X_rule_values, thresholds,
                                         direction_path)

            rules_dict[rule_filter_name] = rule_filter_srs

        # Keep on traversing until the index is negative (leaf node).
        # Make copies of rule path to send to left and right children.
        # This is necessary since otherwise the recursive calls will
        # overwrite the original one.
        if left_child_index >= 0:
            rule_path_left = rule_path.copy()
            direction_path_left = (direction_path + ['left']).copy()

            _traverse(left_child_index, rule_path_left, direction_path_left)

        if right_child_index >= 0:
            rule_path_right = rule_path.copy()
            direction_path_right = (direction_path + ['right']).copy()

            _traverse(right_child_index, rule_path_right, direction_path_right)

    def _create_rule_filter(X_rule_values, thresholds, direction_path):
        """Creates a boolean Series that filters according to the rule."""
        def _create_single_filter(srs, threshold, direction):
            """Creates a filter for a single node."""
            if direction == 'left':
                return srs <= threshold
            elif direction == 'right':
                return srs > threshold


        # Initialize as True in order to apply '&' operator in for loop
        rule_filter_srs = True

        # direction_path is one shorter than thresholds and the number
        # of columns in X_rule_values
        for i in range(len(direction_path)):
            single_filter = _create_single_filter(X_rule_values.iloc[:, i],
                                                  thresholds[i],
                                                  direction_path[i]
                                                 )
            rule_filter_srs = rule_filter_srs & single_filter

        return rule_filter_srs

    def _create_rule_filter_name(X_rule_values, thresholds, direction_path):
        """Creates the name for the rule filter path."""
        def _create_single_filter_name(i):
            """Creates a single filter name."""
            col_name = X_rule_values.columns[i]
            threshold_val = thresholds[i]

            if direction_path[i] == 'left':
                direction_path_op = '<='
            elif direction_path[i] == 'right':
                direction_path_op = '>'

            filter_name = f'{col_name} {direction_path_op} {threshold_val}'

            return filter_name

        def _combine_filter_names(name_1, name_2):
            """Combine two filter names."""
            return f'{name_1} & {name_2}'


        # Create filter name for each node along the path
        filter_names = [_create_single_filter_name(i)
                            for i in range(len(direction_path))]

        # Combine all filter names into a single rule filter
        rule_filter_name = reduce(_combine_filter_names, filter_names)

        return rule_filter_name


    # Lists to store all cumulative rule path and directions
    rule_path = []
    direction_path = []

    # Dict to store final rules
    rules_dict = {}

    # Initiate traversal from root node
    _traverse(0, rule_path, direction_path)

    return rules_dict
