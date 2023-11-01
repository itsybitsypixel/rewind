from rewind import *

def find_potential_root_nodes(xbr: XBEBinaryReader):
    potential_node_virtual_addresses = []
    referenced_node_virtual_addresses = []

    for i, section in enumerate(xbr.xbe.sections):
        virtual_address = section.vaddr

        print(f'Searching in section {i+1} of {len(xbr.xbe.sections)} at virtual address {virtual_address:#0{10}x}...')
        while virtual_address < (section.vaddr + section.rsize) - 0x34:
            potential_node_type = xbr.read_i32(virtual_address)
            #print(virtual_address)

            # Scale is often (1.0, 1.0, 1.0) on root nodes and has a node type between 0x00 and 0xff, so we'll search
            # for that pattern.
            if ((potential_node_type > 0x00 and potential_node_type < 0xff) and
                xbr.read_f32(virtual_address + 0x20) == 1.0 and
                xbr.read_f32(virtual_address + 0x24) == 1.0 and
                xbr.read_f32(virtual_address + 0x28) == 1.0):

                referenced_node_virtual_addresses.append(xbr.read_i32(virtual_address + 0x2c))
                referenced_node_virtual_addresses.append(xbr.read_i32(virtual_address + 0x30))

                potential_node_virtual_addresses.append(virtual_address)
            virtual_address += 4

    # Eliminate potential nodes that are referenced by other nodes, as they are either children or siblings and not
    # root nodes.
    root_node_virtual_addresses = []
    for node_virtual_address in potential_node_virtual_addresses:
        if node_virtual_address not in referenced_node_virtual_addresses:
            root_node_virtual_addresses.append(node_virtual_address)

    return root_node_virtual_addresses
    
if __name__ == '__main__':
    supported_formats = [
        'gltf',
        'glb'
    ]

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('xbe_filename')
    args_parser.add_argument('-t', '--texture-directory',
                             help = 'directory where game textures are stored',
                             required = True)
    
    args_parser.add_argument('-f', '--format',
                             help = 'output format [' + ', '.join(supported_formats) + ']',
                             default = 'glb')
    
    args_parser.add_argument('-o', '--output-directory',
                             required = True)
    
    args = args_parser.parse_args()

    args.format = args.format.lower()
    if args.format not in supported_formats:
        print(f'Unsupported output format \'{args.format}\', supported formats are [' + ', '.join(supported_formats) + ']')
        sys.exit(1)

    xbe = XBE.from_filepath(args.xbe_filename)
    potential_root_nodes = find_potential_root_nodes(XBEBinaryReader(xbe))

    for i, potential_root_node_virtual_address in enumerate(potential_root_nodes):
        print(f'{i + 1}/{len(potential_root_nodes)} - {potential_root_node_virtual_address:#0{10}x}')

        ctx = RewindContext(xbe)
        try:
            ctx.parse_node(potential_root_node_virtual_address)
        except UnrecognizedChunkIdentifierError as e:
            print(e.message)
            continue
        except:
            continue

        ctx.parse_texture_list(None, args.texture_directory, 32)

        save_as_gltf(ctx, args.output_directory, args.format, None)

